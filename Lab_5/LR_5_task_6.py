import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Генерація тренувальних даних
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 5 * np.square(x) + 1
y /= np.linalg.norm(y)

# Створення даних та міток
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# Побудова графіка вхідних даних
plt.figure()
plt.scatter(data, labels)
plt.xlabel("Розмірність 1")
plt.ylabel("Розмірність 2")
plt.title("Вхідні дані")

# Визначення багатошарової нейронної мережі з двома прихованими
# шарами. Перший прихований шар складається із чотирьох нейронів.
# Другий прихований шар складається з одного нейрону.
# Вихідний шар складається з одного нейрона.
nn = nl.net.newff([[min_val, max_val]], [4,1])


# Завдання градієнтного спуску як навчального алгоритму
nn.trainf = nl.train.train_gd

# Тренування нейронної мережі
error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

# Виконання нейронної мережі на тренувальних даних
output = nn.sim(data)
y_pred = output.reshape(num_points)

# Побудова графіка помилки навчання
plt.figure()
plt.plot(error_progress)
plt.xlabel('Кількість епох')
plt.ylabel('Помилка навчання')
plt.title('Зміна помилку навчання')

# Побудова графіка результатів
x_dende = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dende.reshape(x_dende.size,1)).reshape(x_dende.size)
plt.figure()
plt.plot(x_dende, y_dense_pred, '-', x, y, '0.', x, y_pred, 'p')
plt.title('Фактичні та прогнозовані значення')
plt.show()