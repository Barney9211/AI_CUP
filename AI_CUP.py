from pathlib import Path, PurePath
import numpy as np, pandas as pd, lightgbm as lgb, optuna, warnings, json
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import logit, expit
warnings.filterwarnings('ignore')
def logit_mean(p):
    return float(expit(logit(np.clip(p, 1e-6, 1-1e-6)).mean()))

def prob_binary(pred, want_one=True):
    if pred.ndim == 1:
        return pred if want_one else 1.0 - pred
    return pred[:, 1] if want_one else pred[:, 0]

def suggest_params(trial, objective, num_class=None):
    params = dict(
        boosting_type   = 'gbdt',
        objective       = objective,
        metric          = 'binary_logloss' if objective=='binary' else 'multi_logloss',
        learning_rate   = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        num_leaves      = trial.suggest_int('num_leaves', 31, 127),
        max_depth       = trial.suggest_int('max_depth', -1, 10),
        min_child_samples = trial.suggest_int('min_child_samples', 20, 100),
        feature_fraction = trial.suggest_float('feature_fraction', 0.6, 0.9),
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.6, 0.9),
        bagging_freq      = trial.suggest_int('bagging_freq', 1, 5),
        lambda_l1       = trial.suggest_float('lambda_l1', 0.0, 1.0),
        lambda_l2       = trial.suggest_float('lambda_l2', 0.0, 1.0),
        seed            = 42,
        verbose         = -1
    )
    if objective == 'binary':
        params['is_unbalance'] = True
    else:
        params['num_class'] = num_class
    return params

# ---------- Optuna ----------
def logloss_objective(X, y, groups, objective, num_class=None, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    def _obj(trial):
        params  = suggest_params(trial, objective, num_class)
        n_round = max(300, min(int(800/params['learning_rate']), 4000))
        losses  = []
        #訓練完 然後算驗證集的loss
        for tr_idx, val_idx in sgkf.split(X, y, groups):
            bst = lgb.train(
                params,
                lgb.Dataset(X[tr_idx], y[tr_idx]),
                num_boost_round=n_round,
                valid_sets=[lgb.Dataset(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            pred = bst.predict(X[val_idx])
            loss = log_loss(y[val_idx], pred,
                            labels=[0,1] if objective=='binary'
                            else list(range(num_class)))
            losses.append(loss)
        return np.mean(losses)
    return _obj

# ---------- 載入資料 ----------
def load_data():
    info = pd.read_csv(PurePath('39_Training_Dataset','train_info.csv'))
    tgts = ['gender','hold racket handed','play years','level']
    Xs, groups, y_map = [], [], {t:[] for t in tgts}
    for p in Path('./tabular_data_train').glob('*.csv'):
        uid  = int(p.stem)
        meta = info[info['unique_id']==uid]
        if meta.empty: continue
        pid  = meta['player_id'].iat[0]
        arr  = pd.read_csv(p).values
        Xs.append(arr)
        for t in tgts:
            y_map[t].extend([meta[t].iat[0]]*len(arr))
        groups.extend([pid]*len(arr))
    scaler = MinMaxScaler().fit(np.vstack(Xs))
    X  = scaler.transform(np.vstack(Xs))
    # print(X.shape, len(groups))
    enc= {}
    for t in tgts:
        le = LabelEncoder().fit(y_map[t])
        y_map[t] = le.transform(y_map[t])
        enc[t] = le
    return X, y_map, np.array(groups), scaler, enc

def train_kfold_models(X, y, groups, best_params, K=5):
    models = []
    sgkf   = StratifiedGroupKFold(K, shuffle=True, random_state=42)
    for tr_idx, val_idx in sgkf.split(X, y, groups):
        bst = lgb.train(
            best_params,
            lgb.Dataset(X[tr_idx], y[tr_idx]),
            num_boost_round=4000,
            valid_sets=[lgb.Dataset(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        models.append(bst)
    return models

# ---------- 主程式 ----------
def main():
    X, y, g, scaler, enc = load_data()

    def fit_ensemble(tag, obj, nc=None):
        # 1. #40個trial 每個trial有自己的超參數 跑獨立的五組模型 每個模型有自己的4組train 1組val 然後最後看哪一份超參數表現最佳(依據:log-loss)
        study = optuna.create_study(direction='minimize')
        study.optimize(logloss_objective(X, y[tag], g, obj, nc),
                       n_trials=40, timeout=1800)
        bestp = suggest_params(optuna.trial.FixedTrial(study.best_params), obj, nc)
        # 2. 用最佳參數做 5-fold OOF 模型
        models = train_kfold_models(X, y[tag], g, bestp, K=5)
        return models, study.best_params

    m_g, bp_g = fit_ensemble('gender', 'binary')
    m_h, bp_h = fit_ensemble('hold racket handed', 'binary')
    m_y, bp_y = fit_ensemble('play years', 'multiclass',
                             len(enc['play years'].classes_))
    m_l, bp_l = fit_ensemble('level', 'multiclass',
                             len(enc['level'].classes_))
    print("四個任務的 5-fold 模型訓練完成")
    # ---------- 推論 ----------
    gs = 27
    def avg_pred(models, Xf):
        return np.mean([m.predict(Xf) for m in models], axis=0)
    submission = []
    for p in sorted(Path('./tabular_data_test').glob('*.csv'),key=lambda q:int(q.stem)):
        uid = int(p.stem)
        Xf  = scaler.transform(pd.read_csv(p).values)
        g_raw = prob_binary(avg_pred(m_g, Xf), want_one=False)   # 男機率
        h_raw = prob_binary(avg_pred(m_h, Xf), want_one=False)   # 右手機率
        years_prob = avg_pred(m_y, Xf).mean(axis=0)
        level_prob = avg_pred(m_l, Xf).mean(axis=0)
        submission.append({
            'unique_id'         : uid,
            'gender'            : logit_mean(g_raw),
            'hold racket handed': logit_mean(h_raw),
            'play years_0'      : years_prob[0],
            'play years_1'      : years_prob[1],
            'play years_2'      : years_prob[2],
            **{f'level_{lab}': level_prob[i]
               for i, lab in enumerate(enc['level'].classes_)}
        })
    pd.DataFrame(submission).round(4).to_csv('LBM_logloss_oof.csv', index=False)
    print("LBM_logloss_oof.csv")
    with open('best_params_logloss_oof.json','w',encoding='utf-8') as f:
        json.dump({'gender':bp_g,'hold':bp_h,'years':bp_y,'level':bp_l},
                  f,indent=2,ensure_ascii=False)
if __name__ == '__main__':
    main()
