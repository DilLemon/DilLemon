from statistics import quantiles
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data2 = pd.read_csv('datalist1.csv')
df = pd.read_csv('datalist1.csv', usecols = ['Time','Pokazania'])


data = pd.read_csv('datalist.csv')
print(data)

# Общее кол-во строк
num_rows = len(data)
print("Количество строк:", num_rows)

# Кол-во уникальных дат в датасете
num_unique_dates = data['Date'].nunique()
print("Количество уникальных дат:", num_unique_dates)

# Выявление выбросов 
q1 = data['Pokazania'].quantile(0.25)
q3 = data['Pokazania'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data['Pokazania'] < lower_bound) | (data['Pokazania'] > upper_bound)]
print("Количество выбросов:", len(outliers))

# Проверка на пустые значения
print("Пустые значения: ", "\n", data.isnull().sum())

# Проверка на дублирование значений
print("Дублированные значения: ", data.duplicated().sum())

# Построение графика распределения показаний счётчика
plt.figure(figsize=(10, 6))
sns.histplot(data['Pokazania'], kde=True)
plt.title("Распределение показаний счётчика")
plt.xlabel("Показания")
plt.ylabel("Частота")
plt.show()

# Построение графика распределения показаний счётчика (без выбросов)
plt.figure(figsize=(10, 6))
sns.histplot(data['Pokazania'][~((data['Pokazania'] < lower_bound) | (data['Pokazania'] > upper_bound))], kde=True)
plt.title("Распределение показаний счётчика (без выбросов)")
plt.xlabel("Показания")
plt.ylabel("Частота")
plt.show()



# 6) Корреляционный анализ числовых данных
numeric_data = data2[['Week Day','Pokazania','Time']]
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='flare')
plt.title("Матрица корреляций")
plt.show()

median = df.Pokazania.median()
IQR = df.Pokazania.quantile(0.75) - df.Pokazania.quantile(0.25)
perc25 = df.Pokazania.quantile(0.25)
perc75 = df.Pokazania.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25),'75-й перцентиль: {},'.format(perc75), "IQR: {}, ".format(IQR),"Границы выбросов:[{f}, {l}].".format(f=perc25 - 1.5*IQR,l=perc75 + 1.5*IQR))
df.Pokazania.loc[df.Pokazania.between(perc25 - 1.5*IQR,perc75 + 1.5*IQR)].hist(bins = 10,range=(0,10),label = 'IQR')
plt.legend();

df = df.loc[df.Pokazania.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)]
df.Pokazania.describe()

X = df[['Time']]
Y = df['Pokazania']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, Y_train, Y_test = train_test_split(df['Time'], df['Pokazania'], test_size=0.2, random_state=42)

# Построение и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), Y_train.values)
y_pred_train = model.predict(X_train.values.reshape(-1, 1))
y_pred_test = model.predict(X_test.values.reshape(-1, 1))

# Визуализация результатов
plt.scatter(X_train, Y_train, s = 7, color='blue', label='Train')
plt.scatter(X_test, Y_test, s = 7, color='red', label='Test')
plt.plot(X_train, y_pred_train, color='black', label='Linear Regression')
plt.xlabel('Time')
plt.ylabel('Pokazania')
plt.legend()
plt.title('График линейной регрессии')
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X1_Poly = poly.fit_transform(X)

mod = LinearRegression()
mod.fit(X1_Poly, Y)

ypred = mod.predict(X1_Poly)
plt.scatter(X, Y, s=6 ,color = 'blue',label = 'Данные')
plt.plot(X,ypred, color ='red', linewidth = 2, label = 'Полиномиальная регрессия')
plt.xlabel('Time')
plt.ylabel('Pokazania')
plt.legend()
plt.title('График полиномиальной регрессии')
plt.show()



# MAE
print('MAE:', metrics.mean_absolute_error(Y_test,y_pred_test))
# MSE
print('MSE:', metrics.mean_squared_error(Y_test,y_pred_test))
#Вычисляем коэффициент детерминации
print('R_2:', metrics.r2_score(Y_test, y_pred_test))
