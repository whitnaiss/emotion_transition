import matplotlib.pyplot as plt
import numpy as np

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
models = ['qwen', 'qwen + emotion transition', 'llama', 'llama + emotion transition', 'bert', 'bert + emotion transition']
data = [
    [0.985, 0.9850, 0.985, 0.985],
    [0.990, 0.9902, 0.990, 0.990],
    [0.990, 0.9900, 0.990, 0.990],
    [0.990, 0.9900, 0.990, 0.990],
    [0.91, 0.9234, 0.91, 0.909],
    [0.99, 0.99, 0.99, 0.99]
]

x = np.arange(len(labels))
width = 0.12  # narrower bars for more models
colors = ['#2878B5', '#C82423', '#F8AC8C', '#7A6FBE', '#26A69A', '#FFB300']
plt.figure(figsize=(12, 6))

for i, (model, scores) in enumerate(zip(models, data)):
    bars = plt.bar(x + width*i - width*2.5, scores, width, label=model, color=colors[i % len(colors)])
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, height + 0.003, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.ylim(0.85, 1.03)
plt.ylabel('Score')
plt.title('Comparison of Model Metrics')
plt.xticks(x, labels)
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout(rect=[0,0,0.85,1])
# plt.show()
plt.savefig('./model_compare.png') 