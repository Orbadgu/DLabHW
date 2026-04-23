import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Target'] = (df['Spending Score (1-100)'] >= 50).astype(int)
    df.drop(['CustomerID', 'Spending Score (1-100)'], axis=1, inplace=True)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    X = df.drop('Target', axis=1).values
    y = df['Target'].values.reshape(-1, 1)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_temp, X_test, y_temp, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_dev_scaled, y_dev, X_test_scaled, y_test
