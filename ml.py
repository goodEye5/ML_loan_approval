import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

#veri setinin yüklenmesi
path = r"C:\veri\loan_approval_dataset.xlsx"
veriseti=pd.read_excel(path)

#tanımlayıcı istatistikler
veriseti.info()
print(veriseti.describe())
print(veriseti.describe(include=['O']))
veriseti = veriseti.drop('loan_id', axis=1)


#pasta grafiği
fig, axes = plt.subplots(1,3, figsize=(18,6))

print(veriseti["self_employed"].value_counts(0))
print(veriseti["education"].value_counts(1))
print(veriseti["loan_status"].value_counts(2))

veriseti["self_employed"].value_counts(0).plot(kind='pie',autopct="%.2f%%",ax=axes[0] ,title="Kendi işi mi?",startangle=90, pctdistance=0.85, textprops={'fontsize': 12}, colors=['#ff9999','#66b3ff'], labeldistance=1.1)
veriseti["education"].value_counts(1).plot(kind='pie',autopct="%.2f%%",ax=axes[1],title="Eğitim Durumu",startangle=90, pctdistance=0.85, textprops={'fontsize': 12}, colors=['#ff9999','#66b3ff'], labeldistance=1.1)
veriseti["loan_status"].value_counts(2).plot(kind='pie',autopct="%.2f%%",ax=axes[2],title="Kredi Sonucu",startangle=90, pctdistance=0.85, textprops={'fontsize': 12}, colors=['#ff9999','#66b3ff'], labeldistance=1.1)
plt.tight_layout()
plt.show()

#korelasyon analizi
plt.figure(figsize=(10,8))
veriseti_numeric=veriseti.drop(["self_employed","education","loan_status"],axis=1)
sb.heatmap(veriseti_numeric.corr(),annot=True,fmt=".3f",cmap="coolwarm",linewidths=0.5,cbar_kws={'label': 'Korelasyon Katsayısı'}, annot_kws={'size': 10}, xticklabels=veriseti_numeric.columns, yticklabels=veriseti_numeric.columns)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title("Korelasyon Isı Haritası", fontsize=14)
plt.tight_layout()
plt.show()


#eksik veri düzenlenmesi
print(veriseti.isnull().sum().sort_values(ascending=False))
XY=veriseti.iloc[:,2:12].values
yaklasikdeger=SimpleImputer(missing_values=np.nan,strategy='mean')
XY = yaklasikdeger.fit_transform(XY)
print(yaklasikdeger.statistics_)

#Kategorik verilerin dönüştürülmesi
XK =veriseti.iloc[:,0:2].values
ct=ColumnTransformer([("education",OneHotEncoder(sparse_output=False),[0]),("self_employed",OneHotEncoder(sparse_output=False),[1])],remainder='passthrough')
XK =ct.fit_transform(XK)
yeni_veriseti=np.concatenate((XK,XY),axis=1)

#bağımlı ve bağımsız değişkenkenlerin veri setinden çekilmesi
x= yeni_veriseti[:,:13]
y= yeni_veriseti[:,13]

#öznitelik ölçeklendirme:minmaxscaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)

#Öznitelik Seçimi:Filtre=Anova
selector = SelectKBest(score_func=f_classif, k=6)
selector.fit(x, y)
x_selected = selector.transform(x)

print("---Filtre ANOVA----")
selected_features = selector.get_support(indices=True)
print("Seçilen özniteliklerin indeksleri:",selected_features)



#Eğitim ve test veri setlerinin oluşturulması
x_train,x_test,y_train,y_test=train_test_split(x_selected,y,test_size=0.3,random_state=0)



#Karar Ağacı
model_kararagaci = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=10)
model_kararagaci.fit(x_train, y_train)
y_tahminkararagaci = model_kararagaci.predict(x_test)

#K-en yakın komşu
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(x_train,y_train)
y_tahminknn = model_knn.predict(x_test)

#Naive Bayes
model_naivebayes = GaussianNB()
model_naivebayes.fit(x_train,y_train)
y_tahminnaivebayes = model_naivebayes.predict(x_test)

#Yapay sinir ağları
model_ysa = MLPClassifier(hidden_layer_sizes=(10,10,10),activation ='relu', tol=0.001, max_iter=10000, random_state=0)
history=model_ysa.fit(x_train,y_train)
y_tahminysa = model_ysa.predict(x_test)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.loss_curve_)
plt.title('Öğrenme Eğrisi')
plt.xlabel('İterasyon')
plt.ylabel('Hata Değeri')

#Ysa Parametre Optimizsasyonu
model_ysa_hiper= MLPClassifier(max_iter=10000)
parametre_aralik = {'hidden_layer_sizes': [(10,),(30,),(50,),(10,10),(10,10,10)], 'activation': ['relu','tanh','logistic'], 'alpha':[0.001,0.01], 'learning_rate': ['constant','invscaling','adaptive']}

grid_search = GridSearchCV(model_ysa_hiper,parametre_aralik,scoring='neg_mean_absolute_error',cv=5)
grid_search.fit(x_train,y_train)
print("En İyi Hiperparametreler:",grid_search.best_params_)
en_iyi_model = grid_search.best_estimator_
y_tahminhiper = en_iyi_model.predict(x_test)



#Doğruluk Oranı ve Performans Ölçütü Değerleri
print("Kararağacı Doğruluk Oranı:", accuracy_score(y_test, y_tahminkararagaci))
print("\nPerformans Ölçütleri:\n", classification_report(y_test, y_tahminkararagaci))
print("------------------------------------------------")

print("Knn Doğruluk Oranı:", accuracy_score(y_test, y_tahminknn))
print("\nPerformans Ölçütleri:\n", classification_report(y_test, y_tahminknn))
print("------------------------------------------------")

print("Naivebayes Doğruluk Oranı:", accuracy_score(y_test, y_tahminnaivebayes))
print("\nPerformans Ölçütleri:\n", classification_report(y_test, y_tahminnaivebayes))
print("------------------------------------------------")

print("Ysa Doğruluk Oranı:", accuracy_score(y_test, y_tahminysa))
print("\nPerformans Ölçütleri:\n", classification_report(y_test, y_tahminysa))
print("------------------------------------------------")

print("ysahiperAccuracy:", accuracy_score(y_test, y_tahminhiper))
print("\nClassification Report:\n", classification_report(y_test, y_tahminhiper))

#Eğitim ve test doğruluklarının karşılaştırılmasıyla çapraz Doğrulama Skorları (overfitting-underfitting kontrolü)
train_accuracy = accuracy_score(y_train, model_kararagaci.predict(x_train))
test_accuracy = accuracy_score(y_test, y_tahminkararagaci)
print("Karar Ağacı")
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


cv_scores = cross_val_score(model_kararagaci, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("CV Std Dev:", np.std(cv_scores))
print()


train_accuracy = accuracy_score(y_train, model_knn.predict(x_train))
test_accuracy = accuracy_score(y_test, y_tahminknn)
print("Knn")
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


cv_scores = cross_val_score(model_knn, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("CV Std Dev:", np.std(cv_scores))
print()


train_accuracy = accuracy_score(y_train, model_naivebayes.predict(x_train))
test_accuracy = accuracy_score(y_test, y_tahminnaivebayes)
print("Naive Bayes")
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


cv_scores = cross_val_score(model_naivebayes, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("CV Std Dev:", np.std(cv_scores))
print()


train_accuracy = accuracy_score(y_train, model_ysa.predict(x_train))
test_accuracy = accuracy_score(y_test, y_tahminysa)
print("Yapay Sinir Ağları")
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


cv_scores = cross_val_score(model_ysa, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("CV Std Dev:", np.std(cv_scores))
print()


train_accuracy = accuracy_score(y_train, en_iyi_model.predict(x_train))
test_accuracy = accuracy_score(y_test, y_tahminhiper)
print("ysahiperTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


cv_scores = cross_val_score(model_ysa_hiper, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("CV Std Dev:", np.std(cv_scores))


#kodu kısaltmak için results adı altında birleştirme
results = {
    "Decision Tree": {
        "model": model_kararagaci,
        "y_pred": y_tahminkararagaci,
        "accuracy": accuracy_score(y_test, y_tahminkararagaci),
        "classification_report": classification_report(y_test, y_tahminkararagaci, output_dict=True),
        "cv_scores": cross_val_score(model_kararagaci, x_train, y_train, cv=5)
    },
    "KNN": {
        "model": model_knn,
        "y_pred": y_tahminknn,
        "accuracy": accuracy_score(y_test, y_tahminknn),
        "classification_report": classification_report(y_test, y_tahminknn, output_dict=True),
        "cv_scores": cross_val_score(model_knn, x_train, y_train, cv=5)
    },
    "Naive Bayes": {
        "model": model_naivebayes,
        "y_pred": y_tahminnaivebayes,
        "accuracy": accuracy_score(y_test, y_tahminnaivebayes),
        "classification_report": classification_report(y_test, y_tahminnaivebayes, output_dict=True),
        "cv_scores": cross_val_score(model_naivebayes, x_train, y_train, cv=5)
    },
    "Neural Network": {
        "model": model_ysa,
        "y_pred": y_tahminysa,
        "accuracy": accuracy_score(y_test, y_tahminysa),
        "classification_report": classification_report(y_test, y_tahminysa, output_dict=True),
        "cv_scores": cross_val_score(model_ysa, x_train, y_train, cv=5)
    }
}


#Doğruluk Oranı Kıyaslama Sütun Grafikleri
accuracies = [results[model]["accuracy"] for model in results]
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), accuracies, color=['blue', 'orange', 'green', 'purple'])
plt.title("Model Doğruluk Oranı Karşılaştırması")
plt.ylabel("Doğruluk Oranı")
plt.ylim(0.90, 1)
plt.show()


# Çapraz Doğrulama Kıyaslama Kutu Grafikleri
cv_scores_data = [result["cv_scores"] for result in results.values()]
plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores_data, labels=results.keys(), patch_artist=True, 
            boxprops=dict(facecolor="lightblue", color="blue"), 
            medianprops=dict(color="red"))
plt.title("Çapraz Doğrulama Skoru Dağılımı")
plt.ylabel("Doğruluk Oranı")
plt.show()

#Perfomans Ölçütleri Kıyaslaması
unique_classes = np.unique(y_test)
class_of_interest = unique_classes[1]  


class_label = str(class_of_interest) if str(class_of_interest) in results["Decision Tree"]["classification_report"] else class_of_interest


metrics = ["precision", "recall", "f1-score"]
metric_values = {metric: [] for metric in metrics}

for model_name, result in results.items():
    for metric in metrics:
        metric_values[metric].append(result["classification_report"][class_label][metric])


z = np.arange(len(results.keys()))
bar_width = 0.2

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    plt.bar(z + i * bar_width, metric_values[metric], width=bar_width, label=metric)

plt.xticks(z + bar_width, results.keys(), rotation=45)
plt.ylabel("Skor")
plt.title(f"Performans Ölçütleri Kıyaslaması (Class: {class_label})")
plt.ylim(0.90, 1)
plt.legend(title="Ölçütler")
plt.tight_layout()
plt.show()


















