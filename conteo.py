import pandas as pd

# Cargar los datos procesados
df = pd.read_csv('./data/data_ventas.csv')

# Convertir la fecha a formato datetime
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Agregar columna de semana para agrupar ventas semanales
df['week'] = df['date'].dt.to_period('W')

# Agrupar por producto, ciudad y semana para obtener ventas semanales
df_weekly = df.groupby(['itemdescr', 'areacity', 'week'])['qty'].sum().reset_index()

# Contar cuántas semanas tiene cada combinación producto-ciudad
data_counts = df_weekly.groupby(['itemdescr', 'areacity'])['week'].nunique().reset_index(name='num_semanas')

# 🔹 Calcular semanas totales en el período de cada producto/ciudad
total_weeks = df_weekly.groupby(['itemdescr', 'areacity'])['week'].agg(lambda x: (x.max() - x.min()).n + 1).reset_index(name='total_weeks')

# 🔹 Calcular semanas con ventas
weeks_with_sales = df_weekly.groupby(['itemdescr', 'areacity'])['week'].nunique().reset_index(name='weeks_with_sales')

# 🔹 Unir estos valores al DataFrame principal
df_analysis = data_counts.merge(total_weeks, on=['itemdescr', 'areacity'], how='left')
df_analysis = df_analysis.merge(weeks_with_sales, on=['itemdescr', 'areacity'], how='left')

# 🔹 Calcular porcentaje de semanas sin ventas
df_analysis['porcentaje_semanas_sin_ventas'] = 1 - (df_analysis['weeks_with_sales'] / df_analysis['total_weeks'])

# 🔹 Determinar si el producto es viable
df_analysis['viable'] = df_analysis['porcentaje_semanas_sin_ventas'].apply(lambda x: 'Sí' if x < 0.3 else 'No')

# 🔹 Guardar todos los datos con la clasificación en CSV
df_analysis.sort_values(by='num_semanas', ascending=False).to_csv('productos_analizados.csv', index=False)

# 🔹 Filtrar y guardar solo los productos viables en otro CSV
df_analysis[df_analysis['viable'] == 'Sí'].sort_values(by='num_semanas', ascending=False).to_csv('productos_viables.csv', index=False)

# 🔹 Mostrar en pantalla los productos viables, ordenados de mayor a menor
print(df_analysis[df_analysis['viable'] == 'Sí'].sort_values(by='num_semanas', ascending=False))
