import data_prep
# 'segment' sütununu y olarak ayıralım
y = new_df['segment']

X = new_df.drop(['segment', 'index', 'master_id'], axis=1)
# Sayısal ve kategorik sütunları ayıralım
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Sayısal sütunlar için RobustScaler, kategorik sütunlar için OneHotEncoder kullanarak bir ColumnTransformer oluşturalım
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Pipeline'ı oluşturalım
# Preprocessing ve modelinizi birleştirelim
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # KNN algoritması kullanıyorum, n_neighbors sayısını ihtiyacınıza göre ayarlayabilirsiniz
])

# Veriyi eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline'ı kullanarak sayısal ve kategorik özellikleri işleyip modeli eğitelim
pipeline.fit(X_train, y_train)

# Eğitim setinde modelin performansını değerlendirelim
train_accuracy = pipeline.score(X_train, y_train)
print(f"Eğitim seti doğruluk oranı: {train_accuracy:.2f}")

# Test setinde modelin performansını değerlendirelim
test_accuracy = pipeline.score(X_test, y_test)
print(f"Test seti doğruluk oranı: {test_accuracy:.2f}")

# Modelin tahminlerini alalım
y_pred = pipeline.predict(X_test)

# Confusion Matrix'i hesaplayalım
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report (Precision, Recall, F1-Score) hesaplayalım
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)