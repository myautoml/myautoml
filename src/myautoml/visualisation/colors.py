from matplotlib import cm

TRAIN_COLORS = cm.get_cmap('Pastel1')
TEST_COLORS = cm.get_cmap('Set1')

EVALUATION_COLORS = {
    'train': 'lightsteelblue',
    'valid': 'steelblue',
    'test': 'navy',
    'baseline': 'lightgray'
}

TRAIN_COLOR = EVALUATION_COLORS['train']
VALID_COLOR = EVALUATION_COLORS['valid']
TEST_COLOR = EVALUATION_COLORS['test']
BASELINE_COLOR = EVALUATION_COLORS['baseline']
