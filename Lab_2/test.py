from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей: {}".format(iris_dataset['target_names']))

print("Назва ознак: \n{}".format(iris_dataset['feature_names']))

print("Тип масиву data: {}".format(type(iris_dataset['data'])))

print("Форма масиву data: {}".format(iris_dataset['data'].shape))

print("Тип масиву target: {}".format(type(iris_dataset['target'])))

print("Відповіді:\n{}".format(iris_dataset['target']))


print("Values of the features for the first five examples ")
feature_names = iris_dataset.feature_names
first_five_examples = iris_dataset.data[:5]

for i in range(5):
  print(i + 1)
  for j, feature in enumerate(feature_names):
    print(f"{feature} = {first_five_examples[i][j]}")
  print()