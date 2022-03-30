plot_params = {'savefig.dpi': 600,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

path = {
    'raw': './data/raw/',
    'xray': './data/xray',
    'env': './data/environment',
    'results': './data/results',
    'img': './data/img'
}

tie_points = {
    'sterol_upper': {
        'src': [[89, 92], [244, 22], [324, 19]],
        'dst': [[37, 37], [360, 759], [360, 1132]]
    },
    'sterol_lower': {
        'src': [[61, 30], [150, 102], [306, 39]],
        'dst': [[27, 1161], [358, 1596], [19, 2332]]
    },
    'alkenone_upper': {
        'src': [[88, 88], [235, 25], [307, 91]],
        'dst': [[37, 90], [354, 777], [30, 1123]]
    },
    'alkenone_lower': {
        'src': [[64, 21], [175, 92], [302, 26]],
        'dst': [[30, 1188], [354, 1732], [20, 2332]]
    }
}

calibrants = {
    'alkenone': [557.2523, 533.2523, 553.5319, 537.2261, 564, 535.2104, 522.5972, 550.6285, 561.2472, 559.2316, 573.2472,
           569.4540, 569.2523, 551.2417],
    'sterol': [413.2662, 441.2975, 409.2713, 429.2400, 435.3597, 433.3805, 391.3547, 407.3284, 522.5972, 393.2975]
}