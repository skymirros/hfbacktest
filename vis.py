import optuna
study = optuna.load_study(study_name="mm-custom-30m-2",storage= "mysql://optuna:AyHfbtAyAiRjR4ck@47.86.7.11/optuna")
fig = optuna.visualization.plot_pareto_front(study,target_names=["收益", "平均持仓","回撤"])
fig.show()