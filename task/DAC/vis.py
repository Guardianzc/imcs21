import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Precision, Recall and F1-Score
#                                             precision    recall  f1-score   support
#
#                         Inform-Precautions     0.8397    0.8282    0.8339      1752
#                            Inform-Etiology     0.5521    0.5315    0.5416       508
#                             Inform-Symptom     0.8126    0.8846    0.8471      4481
#                      Inform-Medical_Advice     0.7251    0.7226    0.7238      1157
#                           Request-Etiology     0.6469    0.6946    0.6699       298
#                        Request-Precautions     0.7742    0.7200    0.7461       400
#                                   Diagnose     0.6647    0.7531    0.7061       737
#                  Request-Basic_Information     0.7831    0.7895    0.7863       988
#                                      Other     0.8769    0.8379    0.8569     11214
#  Inform-Existing_Examination_and_Treatment     0.7740    0.7237    0.7480      1987
#                Request-Drug_Recommendation     0.8228    0.8624    0.8422      1061
#                 Inform-Drug_Recommendation     0.7825    0.8390    0.8098      2118
#                            Request-Symptom     0.9104    0.9222    0.9163      3020
# Request-Existing_Examination_and_Treatment     0.9010    0.9078    0.9044      1313
#                     Request-Medical_Advice     0.6525    0.7399    0.6935       373
#                   Inform-Basic_Information     0.6700    0.6152    0.6414      1528
#
#                                   accuracy                         0.8219     32935
#                                  macro avg     0.7618    0.7733    0.7667     32935
#                               weighted avg     0.8228    0.8219    0.8217     32935

# Inform-Precautions
# Inform-Etiology
# Inform-Symptom
# Inform-Medical_Advice
# Request-Etiology
# Request-Precautions
# Diagnose
# Request-Basic_Information
# Other
# Inform-Existing_Examination_and_Treatment
# Request-Drug_Recommendation
# Inform-Drug_Recommendation
# Request-Symptom
# Request-Existing_Examination_and_Treatment
# Request-Medical_Advice
# Inform-Basic_Information

name = ['I-PRCTN', 'I-ETIOL', 'I-SX', 'I-MA', 'R-ETIOL', 'R-PRCTN', 'DIAG', 'I-PC',
        'OTHER', 'I-EET',  'R-DR', 'I-DR', 'R-SX', 'R-EET', 'R-MA', 'R-PC']

cm = np.array([
    [1451, 14, 0, 33, 0, 4, 6, 2, 146, 0, 0, 90, 3, 0, 0, 3],
    [11, 270, 5, 2, 3, 0, 73, 1, 112, 2, 0, 5, 7, 0, 0, 17],
    [1, 3, 3964, 1, 9, 1, 12, 0, 142, 102, 5, 1, 12, 1, 3, 224],
    [38, 8, 1, 836, 0, 0, 18, 1, 148, 1, 0, 97, 1, 7, 1, 0],
    [0, 0, 9, 0, 207, 1, 0, 22, 41, 4, 1, 0, 1, 3, 0, 9],
    [0, 0, 6, 0, 3, 288, 0, 1, 42, 1, 38, 0, 0, 0, 13, 8],
    [5, 35, 1, 8, 0, 0, 555, 2, 103, 12, 0, 8, 3, 5, 0, 0],
    [4, 2, 1, 0, 1, 1, 1, 780, 24, 0, 0, 3, 138, 32, 0, 1],
    [158, 144, 173, 218, 81, 52, 137, 27, 9396, 160, 88, 276, 73, 39, 72, 120],
    [0, 0, 304, 0, 5, 1, 2, 0, 141, 1438, 17, 2, 0, 2, 3, 72],
    [1, 0, 6, 0, 0, 9, 0, 0, 45, 26, 915, 5, 0, 2, 49, 3],
    [56, 5, 0, 44, 0, 0, 21, 0, 195, 3, 1, 1777, 2, 11, 1, 2],
    [3, 4, 9, 4, 4, 0, 7, 126, 43, 0, 0, 2, 2785, 29, 2, 2],
    [0, 4, 0, 6, 4, 0, 3, 34, 26, 2, 5, 3, 33, 1192, 0, 1],
    [0, 0, 5, 1, 0, 7, 0, 0, 36, 6, 40, 1, 0, 0, 276, 1],
    [0, 0, 394, 0, 3, 8, 0, 0, 75, 101, 2, 1, 1, 0, 3, 940]
])


ax = sns.heatmap(cm / cm.sum(axis=0), annot=False, cmap='Blues')
ax.set_xlabel('da')
ax.set_ylabel('da')
# ax.xaxis.set_ticklabels(name)
# ax.yaxis.set_ticklabels(name)
plt.show()

