from model.builders.prostate_models import build_pnet2

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_base = {'id': 'ALL', 'type': 'TFM',
             'params': {
                 'data_type': ['gene_expression'],
                 'drop_AR': False,
                 'cnv_levels': 1,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 2,
             }
             }
data = [data_base]

n_hidden_layers = 5
base_dropout = 0.5
wregs = [0.001] * 7
loss_weights = [2, 7, 20, 54, 148, 400]
wreg_outcomes = [0.01] * 6
pre = {'type': None}

nn_pathway = {
    'type': 'nn',
    'id': 'P-net',
    'params':
        {
            'build_fn': build_pnet2,
            'model_params': {
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference

            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=300,
                                      shuffle=True,
                                      batch_size=10,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=n_hidden_layers + 1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.20, epochs_drop=20),
                                      lr=0.01,
                                      max_f1=True
                                      ),
            'feature_importance': 'deepexplain_deeplift'
        },
}
features = {}
models = [nn_pathway]

class_weight = {0: 0.75, 1: 1.5}
logistic = {'type': 'sgd', 'id': 'Logistic Regression',
            'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01, 'class_weight': class_weight}}
svc = {'type': 'svc',
        'id': 'RBF Support Vector Machine ',
        'params': {'kernel': 'rbf', 'C': 100, 'gamma': 0.001, 'probability': True, 'class_weight': class_weight}}
random_forest = {
        'type': 'random_forest',
        'id': 'Random Forest',
        'params': {'max_depth': None, 'n_estimators': 50, 'bootstrap': False, 'class_weight': class_weight}
    }
adaboost = {
        'type': 'adaboost',
        'id': 'Adaptive Boosting',
        'params': {'learning_rate': 0.1, 'n_estimators': 50}
    }
decision_tree = {
        'type': 'decision_tree',
        'id': 'Decision Tree',
        'params': {'min_samples_split': 10, 'max_depth': 10}
    }
models.append(logistic)
models.append(svc)
models.append(random_forest)
models.append(adaboost)
models.append(decision_tree)

pipeline = {'type': 'one_split', 'params': {'save_train': True, 'eval_dataset': 'test'}}
