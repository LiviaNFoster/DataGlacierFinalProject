## Section for importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/Olivia/Desktop/FinalProject/Healthcare_dataset.csv')

#Taking count of rows and columns ## Livia
row_num = len(df.index) #number of rows
print("Number of rows: ", row_num)
col_num = len(df.columns) #number of columns
print("Number of columns: ", col_num)

#information about the dataframe ## Livia
df.info()

df.head() ##Livia

# Replace Column with a comma in the name ## Livia
df.rename(columns = {'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx':'Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx'},
          inplace = True)

# Finding out how many different outputs there are for our data: ## Livia

#Target Variable:
print("Persistency Flag:\n",df.Persistency_Flag.value_counts())
print("\n")

# Demographics ## Livia
print("Gender:\n",df.Gender.value_counts())
print("\n")

print("Race:\n",df.Race.value_counts())
print("\n")

print("Age:\n",df.Age_Bucket.value_counts())
print("\n")

print("Region:\n",df.Region.value_counts())
print("\n")

print("IDN Indicator:\n",df.Idn_Indicator.value_counts())
print("\n")

# NTM ## Livia
print("Physician's Specialty:\n", df.Ntm_Speciality.value_counts())
print("\n")

print("NTM Specialist Flag:\n",df.Ntm_Specialist_Flag.value_counts())
print("\n")

print("NTM Speciality_Bucket:\n",df.Ntm_Speciality_Bucket.value_counts())
print("\n")

# row 2847
df[df["Ntm_Speciality"] == 'OBSTETRICS & OBSTETRICS & GYNECOLOGY & OBSTETRICS & GYNECOLOGY']

# row 1291
df[df["Ntm_Speciality"] == 'NEUROLOGY']

df[df["Ntm_Speciality"] == 'HEMATOLOGY & ONCOLOGY']

def speciality_converter(col):
    if col == 'ONCOLOGY':
        return 'HEMATOLOGY & ONCOLOGY'
    elif col == 'OBSTETRICS & OBSTETRICS & GYNECOLOGY & OBSTETRICS & GYNECOLOGY':
        return 'OBSTETRICS AND GYNECOLOGY'
    elif col == 'NEUROLOGY':
        return 'PSYCHIATRY AND NEUROLOGY'
    else:
        return col

df['Ntm_Speciality'] = df['Ntm_Speciality'].map(speciality_converter)

print("Physician's Speciality:\n", df.Ntm_Speciality.value_counts()) ##Livia

# all information provided by Physician's specialty column, nothing new
df.drop(['Ntm_Speciality_Bucket'], axis=1, inplace=True)

# Glucocorticoid ## Livia
print("Glucocorticoid Record Prior to NTM:\n",df.Gluco_Record_Prior_Ntm.value_counts())
print("\n")

print("Glucocorticoid Record During to Rx:\n",df.Gluco_Record_During_Rx.value_counts())
print("\n")

# DEXA Scan?? ## Livia
print("DEXA Scan Frequency During Rx:\n",df.Dexa_Freq_During_Rx.value_counts())
print("\n")

print("DEXA Scan During Rx:\n",df.Dexa_During_Rx.value_counts())
print("\n")

df.drop(['Dexa_During_Rx'], axis=1, inplace=True)

# Fragility Fracture
print("Fragility Fracture Prior to NTM:\n",df.Frag_Frac_Prior_Ntm.value_counts())
print("\n")

print("Fragility Fracture During Rx:\n",df.Frag_Frac_During_Rx.value_counts())
print("\n")

# Risk Segment
print("Risk Segment Prior to NTM:\n",df.Risk_Segment_Prior_Ntm.value_counts())
print("\n")

print("Risk Segment During Rx:\n",df.Risk_Segment_During_Rx.value_counts())
print("\n")

print("Change in Risk Segment:\n",df.Change_Risk_Segment.value_counts())
print("\n")

# T Score ## Livia
print("T-score Bucket Prior to NTM:\n",df.Tscore_Bucket_Prior_Ntm.value_counts())
print("\n")

print("T-score Bucket During Rx:\n",df.Tscore_Bucket_During_Rx.value_counts())
print("\n")

print("Change in T-score:\n",df.Change_T_Score.value_counts())
print("\n")

#Comorbidities ## Livia
print("Comorbidities Encountered When Screening For Malignant Neoplasms:\n",
      df.Comorb_Encounter_For_Screening_For_Malignant_Neoplasms.value_counts())
print("\n")

print("Comorbidities Encounterd For Immunization:\n",
      df.Comorb_Encounter_For_Immunization.value_counts())
print("\n")

print("Comorbidities Encounterd For General Exam without Complaint, Suspected or Reported Dx:\n",
      df.Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx.value_counts())
print("\n")

print("Comorbidity Vitamin D Deficiency:\n",
      df.Comorb_Vitamin_D_Deficiency.value_counts())
print("\n")

print("Comorbidity-- Other Joint Disorder Not Elsewhere Classified:\n",
      df.Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified.value_counts())
print("\n")

print("Comorbidity-- Encounter For Other Special(?) Exam without Complaint, Suspected Or Reported Dx:\n",
      df.Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx.value_counts())
print("\n")

print("Comorbidity-- Long-Term Current Drug Therapy:\n",
      df.Comorb_Long_Term_Current_Drug_Therapy.value_counts())
print("\n")

print("Comorbidity-- Dorsalgia:\n",
      df.Comorb_Dorsalgia.value_counts())
print("\n")

print("Comorbidity-- Personal History Of Other Diseases And Conditions:\n",
      df.Comorb_Personal_History_Of_Other_Diseases_And_Conditions.value_counts())
print("\n")

print("Comorbidity-- Other Disorders Of Bone Density And Structure:\n",
      df.Comorb_Other_Disorders_Of_Bone_Density_And_Structure.value_counts())
print("\n")

print("Comorbidity-- Disorders of lipoprotein metabolism and other lipidemias:\n",
      df.Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias.value_counts())
print("\n")

print("Comorbidity-- Osteoporosis without current pathological fracture:\n",
      df.Comorb_Osteoporosis_without_current_pathological_fracture.value_counts())
print("\n")

print("Comorbidity-- Personal history of malignant neoplasm:\n",
      df.Comorb_Personal_history_of_malignant_neoplasm.value_counts())
print("\n")

print("Comorbidity-- Gastro and esophageal reflux disease:\n",
      df.Comorb_Gastro_esophageal_reflux_disease.value_counts())
print("\n")

# Concomitancy ## Livia
print("Concomitancy of Cholesterol and Triglyceride Regulating Preparations:\n",
      df.Concom_Cholesterol_And_Triglyceride_Regulating_Preparations.value_counts())
print("\n")

print("Concomitancy of Narcotics:\n",
      df.Concom_Narcotics.value_counts())
print("\n")

print("Concomitancy of Systemic Corticosteroids Plain:\n",
      df.Concom_Systemic_Corticosteroids_Plain.value_counts())
print("\n")

print("Concomitancy of Anti-Depressants And Mood Stabilisers:\n",
      df.Concom_Anti_Depressants_And_Mood_Stabilisers.value_counts())
print("\n")

print("Concomitancy of Fluoroquinolones:\n",
      df.Concom_Fluoroquinolones.value_counts())
print("\n")

print("Concomitancy of Cephalosporins:\n",
      df.Concom_Cephalosporins.value_counts())
print("\n")

print("Concomitancy of Macrolides And Similar Types:\n",
      df.Concom_Macrolides_And_Similar_Types.value_counts())
print("\n")

print("Concomitancy of Broad Spectrum Penicillins:\n",
      df.Concom_Broad_Spectrum_Penicillins.value_counts())
print("\n")

print("Concomitancy of Anaesthetics General:\n",
      df.Concom_Anaesthetics_General.value_counts())
print("\n")

print("Concomitancy of Viral Vaccines:\n",
      df.Concom_Viral_Vaccines.value_counts())

# Risks ## Livia
print("Risk: Type 1 Insulin Dependent Diabetes:\n",
      df.Risk_Type_1_Insulin_Dependent_Diabetes.value_counts())
print("\n")

print("Risk: Osteogenesis Imperfecta:\n",
      df.Risk_Osteogenesis_Imperfecta.value_counts())
print("\n")

print("Risk: Rheumatoid Arthritis:\n",
      df.Risk_Rheumatoid_Arthritis.value_counts())
print("\n")

print("Risk: Untreated Chronic Hyperthyroidism:\n",
      df.Risk_Untreated_Chronic_Hyperthyroidism.value_counts())
print("\n")

print("Risk: Untreated Chronic Hypogonadism:\n",
      df.Risk_Untreated_Chronic_Hypogonadism.value_counts())
print("\n")

print("Risk: Untreated Early Menopause:\n",
      df.Risk_Untreated_Early_Menopause.value_counts())
print("\n")

print("Risk: Patient Parent Fractured Their Hip:\n",
      df.Risk_Patient_Parent_Fractured_Their_Hip.value_counts())
print("\n")

print("Risk: Smoking Tobacco:\n",
      df.Risk_Smoking_Tobacco.value_counts())
print("\n")

print("Risk: Chronic Malnutrition Or Malabsorption:\n",
      df.Risk_Chronic_Malnutrition_Or_Malabsorption.value_counts())
print("\n")

print("Risk: Chronic Liver Disease:\n",
      df.Risk_Chronic_Liver_Disease.value_counts())
print("\n")

print("Risk: Family History Of Osteoporosis:\n",
      df.Risk_Family_History_Of_Osteoporosis.value_counts())
print("\n")

print("Risk: Low Calcium Intake:\n",
      df.Risk_Low_Calcium_Intake.value_counts())
print("\n")

print("Risk: Vitamin D Insufficiency:\n",
      df.Risk_Vitamin_D_Insufficiency.value_counts())
print("\n")

print("Risk: Poor Health Frailty:\n",
      df.Risk_Poor_Health_Frailty.value_counts())
print("\n")

print("Risk: Excessive Thinness:\n",
      df.Risk_Excessive_Thinness.value_counts())
print("\n")

print("Risk: Hysterectomy Oophorectomy:\n",
      df.Risk_Hysterectomy_Oophorectomy.value_counts())
print("\n")

print("Risk: Estrogen Deficiency:\n",
      df.Risk_Estrogen_Deficiency.value_counts())
print("\n")

print("Risk: Immobilization:\n",
      df.Risk_Immobilization.value_counts())
print("\n")

print("Risk: Recurring Falls:\n",
      df.Risk_Recurring_Falls.value_counts())
print("\n")

#things that could not work in a different section
##Livia

print("Adherence Flag:\n",df.Adherent_Flag.value_counts())
print("\n")


print("Injectable Experience During Rx:\n",df.Injectable_Experience_During_Rx.value_counts())
print("\n")

print("Count of Risks:\n",df.Count_Of_Risks.value_counts())
print("\n")

df.head()

lf = df.copy() #Used for pie chart in Data Viz- Tahsin

cat_cols = list(df.select_dtypes(['object']).columns)
def plot_catcols(x, df):
    df['dummy'] = np.ones(shape = df.shape[0])
    for col in x:
        print(col)
        counts = df[['dummy', col]].groupby([col], as_index = False).count()
        fig, ax = plt.subplots(figsize = (10,4))
        graph = plt.barh(counts[col], counts.dummy) #creating a graph
        plt.xticks(rotation=90)
        plt.title('Counts for ' + col)
        plt.xlabel('count')
        #getting percentages
        total = counts['dummy'].sum()
        percentage = []
        for i in range(counts.shape[0]):
            pct = (counts.dummy[i]/total)*100
            percentage.append(round(pct, 2))
        counts['Percentage'] = percentage
        # plotting the graph with percentages
        i = 0
        for p in graph:
            pct = f'{percentage[i]}%'
            width1, height1 =p.get_width(),p.get_height()
            x1 =p.get_x()+width1
            y1=p.get_y()+height1/2
            ax.annotate(pct,(x1,y1))
            i+=1
        plt.show()
plot_catcols(cat_cols, df)

# Perform one-hot encoding of different columns ## Sammy
df['Risk_Low_Calcium_Intake'] = pd.get_dummies(df['Risk_Low_Calcium_Intake'])['Y']
df['Risk_Vitamin_D_Insufficiency'] = pd.get_dummies(df['Risk_Vitamin_D_Insufficiency'])['Y']
df['Risk_Poor_Health_Frailty'] = pd.get_dummies(df['Risk_Poor_Health_Frailty'])['Y']
df['Risk_Hysterectomy_Oophorectomy'] = pd.get_dummies(df['Risk_Hysterectomy_Oophorectomy'])['Y']
df['Risk_Immobilization'] = pd.get_dummies(df['Risk_Immobilization'])['Y']
df['Risk_Recurring_Falls'] = pd.get_dummies(df['Risk_Recurring_Falls'])['Y']
df['Risk_Rheumatoid_Arthritis'] = pd.get_dummies(df['Risk_Rheumatoid_Arthritis'])['Y']
df['Risk_Untreated_Early_Menopause'] = pd.get_dummies(df['Risk_Untreated_Early_Menopause'])['Y']
df['Risk_Untreated_Chronic_Hypogonadism'] = pd.get_dummies(df['Risk_Untreated_Chronic_Hypogonadism'])['Y']
df['Risk_Untreated_Chronic_Hyperthyroidism'] = pd.get_dummies(df['Risk_Untreated_Chronic_Hyperthyroidism'])['Y']
df['Risk_Type_1_Insulin_Dependent_Diabetes'] = pd.get_dummies(df['Risk_Type_1_Insulin_Dependent_Diabetes'])['Y']
df['Risk_Chronic_Malnutrition_Or_Malabsorption'] = pd.get_dummies(df['Risk_Chronic_Malnutrition_Or_Malabsorption'])['Y']
df['Risk_Chronic_Liver_Disease'] = pd.get_dummies(df['Risk_Chronic_Liver_Disease'])['Y']
df['Risk_Smoking_Tobacco'] = pd.get_dummies(df['Risk_Smoking_Tobacco'])['Y']
df['Risk_Excessive_Thinness'] = pd.get_dummies(df['Risk_Excessive_Thinness'])['Y']
df['Risk_Patient_Parent_Fractured_Their_Hip'] = pd.get_dummies(df['Risk_Patient_Parent_Fractured_Their_Hip'])['Y']
df['Risk_Estrogen_Deficiency'] = pd.get_dummies(df['Risk_Estrogen_Deficiency'])['Y']
df['Risk_Osteogenesis_Imperfecta'] = pd.get_dummies(df['Risk_Osteogenesis_Imperfecta'])['Y']
df['Concom_Cholesterol_And_Triglyceride_Regulating_Preparations'] = pd.get_dummies(df['Concom_Cholesterol_And_Triglyceride_Regulating_Preparations'])['Y']
df['Concom_Narcotics'] = pd.get_dummies(df['Concom_Narcotics'])['Y']
df['Concom_Systemic_Corticosteroids_Plain'] = pd.get_dummies(df['Concom_Systemic_Corticosteroids_Plain'])['Y']
df['Concom_Anti_Depressants_And_Mood_Stabilisers'] = pd.get_dummies(df['Concom_Anti_Depressants_And_Mood_Stabilisers'])['Y']
df['Concom_Fluoroquinolones'] = pd.get_dummies(df['Concom_Fluoroquinolones'])['Y']
df['Concom_Cephalosporins'] = pd.get_dummies(df['Concom_Cephalosporins'])['Y']
df['Concom_Macrolides_And_Similar_Types'] = pd.get_dummies(df['Concom_Macrolides_And_Similar_Types'])['Y']
df['Concom_Broad_Spectrum_Penicillins'] = pd.get_dummies(df['Concom_Broad_Spectrum_Penicillins'])['Y']
df['Concom_Anaesthetics_General'] = pd.get_dummies(df['Concom_Anaesthetics_General'])['Y']
df['Concom_Viral_Vaccines'] = pd.get_dummies(df['Concom_Viral_Vaccines'])['Y']
df['Comorb_Encounter_For_Screening_For_Malignant_Neoplasms'] = pd.get_dummies(df['Comorb_Encounter_For_Screening_For_Malignant_Neoplasms'])['Y']
df['Comorb_Encounter_For_Immunization'] = pd.get_dummies(df['Comorb_Encounter_For_Immunization'])['Y']
df['Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx'] = pd.get_dummies(df['Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx'])['Y']
df['Comorb_Vitamin_D_Deficiency'] = pd.get_dummies(df['Comorb_Vitamin_D_Deficiency'])['Y']
df['Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified'] = pd.get_dummies(df['Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified'])['Y']
df['Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx'] = pd.get_dummies(df['Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx'])['Y']
df['Comorb_Long_Term_Current_Drug_Therapy'] = pd.get_dummies(df['Comorb_Long_Term_Current_Drug_Therapy'])['Y']
df['Comorb_Dorsalgia'] = pd.get_dummies(df['Comorb_Dorsalgia'])['Y']
df['Comorb_Personal_History_Of_Other_Diseases_And_Conditions'] = pd.get_dummies(df['Comorb_Personal_History_Of_Other_Diseases_And_Conditions'])['Y']
df['Comorb_Other_Disorders_Of_Bone_Density_And_Structure'] = pd.get_dummies(df['Comorb_Other_Disorders_Of_Bone_Density_And_Structure'])['Y']
df['Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias'] = pd.get_dummies(df['Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias'])['Y']
df['Comorb_Osteoporosis_without_current_pathological_fracture'] = pd.get_dummies(df['Comorb_Osteoporosis_without_current_pathological_fracture'])['Y']
df['Comorb_Personal_history_of_malignant_neoplasm'] = pd.get_dummies(df['Comorb_Personal_history_of_malignant_neoplasm'])['Y']
df['Comorb_Gastro_esophageal_reflux_disease'] = pd.get_dummies(df['Comorb_Gastro_esophageal_reflux_disease'])['Y']
df['Injectable_Experience_During_Rx'] = pd.get_dummies(df['Injectable_Experience_During_Rx'])['Y']
df['Idn_Indicator'] = pd.get_dummies(df['Idn_Indicator'])['Y']
df['Gluco_Record_Prior_Ntm'] = pd.get_dummies(df['Gluco_Record_Prior_Ntm'])['Y']
df['Gluco_Record_During_Rx'] = pd.get_dummies(df['Gluco_Record_During_Rx'])['Y']
df['Frag_Frac_Prior_Ntm'] = pd.get_dummies(df['Frag_Frac_Prior_Ntm'])['Y']
df['Frag_Frac_During_Rx'] = pd.get_dummies(df['Frag_Frac_During_Rx'])['Y']
df['Persistency_Flag'] = pd.get_dummies(df['Persistency_Flag'])['Persistent']
# 1 = Persistent, 0 = Non-Persistent
df['Adherent_Flag'] = pd.get_dummies(df['Adherent_Flag'])['Adherent']
# 1 = Adherent, 0 = Non-Adherent
df['Ntm_Specialist_Flag'] = pd.get_dummies(df['Ntm_Specialist_Flag'])['Specialist']
# 1 = Specialist, 0 = Others
df['Risk_Segment_Prior_Ntm'] = pd.get_dummies(df['Risk_Segment_Prior_Ntm'])['HR_VHR']
# 1 = High Risk, 0 = Low Risk
df['Tscore_Bucket_Prior_Ntm'] = pd.get_dummies(df['Tscore_Bucket_Prior_Ntm'])['<=-2.5']
# 1 = Low Bone Density, 0 = High Bone Density
df['Gender'] = pd.get_dummies(df['Gender'])['Male']
# NOTE: 0 = Female, 1 = Male
df['Risk_Family_History_Of_Osteoporosis'] = pd.get_dummies(df['Risk_Family_History_Of_Osteoporosis'])['Y']
# NOTE: 0 = NO, 1 = YES

#Changing column names ##Tashin
df.rename(columns={'Age_Bucket':'Age','Ptid':'Patient_ID', 'Ntm_Speciality':'NTM_Physician_Speciality','Ntm_Specialist_Flag':'NTM_Specialist_Flag',
                  'Gluco_Record_Prior_Ntm':'Glucocorticoids_Record_Before_NTM',
                  'Gluco_Record_During_Rx':'Glucocorticoids_Record_During_prescription','Dexa_Freq_During_Rx':'DEXA_Scan_Freq_During_Prescription',
                  'Dexa_During_Rx':'DEXA_Scan_During_Therapy','Frag_Frac_Prior_Ntm':'Fragility_Fracture_Before_NTM',
                  'Frag_Frac_During_Rx':'Fragility_Fracture_During_Therapy','Risk_Segment_Prior_Ntm':'Risk_Before_NTM',
                  'Tscore_Bucket_Prior_Ntm':'Tscore_Before_NTM','Risk_Segment_During_Rx':'Risk_During_Prescription',
                  'Tscore_Bucket_During_Rx':'Tscore_During_Prescription','Change_Risk_Segment':'Risk_Change',
                  'Adherent_Flag':'Adherence','Injectable_Experience_During_Rx':'Injection_Usage_During_Prescription',
                  'Comorb_Encounter_For_Screening_For_Malignant_Neoplasms':'Comorb_Detected_For_Malignant_Neoplasms',
                  'Comorb_Encounter_For_Immunization':'Comorb_Detected_For_Immunization','Count_Of_Risks':'Risk_Count',
                  'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx':'Comorb_Encounter_For_General_Examination_W_O_Complaint_Suspected_or_Reported_Diagnosis',
                  'Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx':'Comorb_Encounter_For_Other_Special_Examination_W_O_Complaint_Suspected_or_Reported_Diagnosis',
                  }, inplace=True)

def f(race, ethnicity):
    if ethnicity == 'Hispanic':
        return 'Hispanic'
    else:
        return race

df['Race'] = df.apply(lambda x: f(x.Race, x.Ethnicity), axis=1)
df['Race'].head(10)
df['Race'].value_counts()
df.drop(['Ethnicity'], axis=1, inplace=True)
df_1 = df.copy()
df.info()
# Checking if risk summation match with risk count ##Tahsin
test = []
risks = df.iloc[:, 46:65].to_numpy()
for i in risks:
    s = i.sum()
    test.append(s)

risk_count = df['Risk_Count']
# print(risks)
np.array_equal(test, risk_count)
df.apply(lambda x: x.astype(str).str.lower())
# transform age column to numeric
import random
def age_converter(age):
    if age == '>75':
        return random.randint(75, 100)
    elif age == '65-75':
        return random.randint(65,75)
    elif age == '55-65':
        return random.randint(55,65)
    elif age == '<55':
        return random.randint(18,55)

races = df['Race']
regions = df['Region']
phys_speciality = df['NTM_Physician_Speciality']
persistence = df['Persistency_Flag']

df['Age'] = df['Age'].apply(age_converter)

means = df.groupby('Race')['Persistency_Flag'].mean()
df['Race'] = df['Race'].map(means)
len(df['Race'])
df['Region'] = df['Region'].map(df.groupby('Region')['Persistency_Flag'].mean())
# target encoding can sometimes result in overfitting, let's try additive smoothing to account for it
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)
df['Race'] = calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=10)
df['Region'] = calc_smooth_mean(df, by='Region', on='Persistency_Flag', m=10)
df['NTM_Physician_Speciality'] = calc_smooth_mean(df, by='Region', on='Persistency_Flag', m=10)

from category_encoders import target_encoder
from sklearn.model_selection import KFold

df_xtrain = pd.DataFrame({'Race':list(races[0:2800])})
df_ytrain = pd.DataFrame({'Persistency_Flag':list(persistence[0:2800])})
df_xtest  = pd.DataFrame({'Race':list(races[2800:])})
add_smoothing_comparison = pd.DataFrame(df['Race'][0:2800])
add_smoothing_comparison.columns = ['Comp w/ Add. smoothing']

kf = KFold(n_splits=4, shuffle=True, random_state=1111)

# Target encoding for training set using K-folding.
train_te = pd.DataFrame()
for tra_idx, val_idx in kf.split(df_xtrain):
    te = target_encoder.TargetEncoder(handle_missing='return_nan',handle_unknown='return_nan')
    te.fit(df_xtrain.iloc[tra_idx],df_ytrain.iloc[tra_idx])
    temp = te.transform(df_xtrain.iloc[val_idx])
    train_te = pd.concat([train_te,temp],axis=0)

train_te.sort_index(inplace=True)
train_te.columns = ['Race_TE w/ K-fold']
train_df_race = pd.concat([df_xtrain,train_te,add_smoothing_comparison,df_ytrain],axis=1)

# Target encoding for test set.
te = target_encoder.TargetEncoder(handle_missing='return_nan',handle_unknown='return_nan') #if level is unknown from input data, return nan.
te.fit(df_xtrain,df_ytrain)
test_te = te.transform(df_xtest)
test_te.columns = ['Race_TE w/ K-fold']
test_df_race = pd.concat([df_xtest,test_te],axis=1)

df_xtrain = pd.DataFrame({'Region':list(regions[0:2800])})
df_ytrain = pd.DataFrame({'Persistency_Flag':list(persistence[0:2800])})
df_xtest  = pd.DataFrame({'Region':list(regions[2800:])})
add_smoothing_comparison = pd.DataFrame(df['Region'][0:2800])
add_smoothing_comparison.columns = ['Comp w/ Add. smoothing']

kf = KFold(n_splits=4, shuffle=True, random_state=1111)

# Target encoding for training set using K-folding.
train_te = pd.DataFrame()
for tra_idx, val_idx in kf.split(df_xtrain):
    te = target_encoder.TargetEncoder(handle_missing='return_nan',handle_unknown='return_nan')
    te.fit(df_xtrain.iloc[tra_idx],df_ytrain.iloc[tra_idx])
    temp = te.transform(df_xtrain.iloc[val_idx])
    train_te = pd.concat([train_te,temp],axis=0)

train_te.sort_index(inplace=True)
train_te.columns = ['Region_TE w/ K-fold']
train_df_region = pd.concat([df_xtrain,train_te,add_smoothing_comparison,df_ytrain],axis=1)

# Target encoding for test set.
te = target_encoder.TargetEncoder(handle_missing='return_nan',handle_unknown='return_nan') #if level is unknown from input data, return nan.
te.fit(df_xtrain,df_ytrain)
test_te = te.transform(df_xtest)
test_te.columns = ['Region_TE w/ K-fold']
test_df_region = pd.concat([df_xtest,test_te],axis=1)

df_xtrain = pd.DataFrame({'NTM_Physician_Speciality':list(phys_speciality[0:2800])})
df_ytrain = pd.DataFrame({'Persistency_Flag':list(persistence[0:2800])})
df_xtest  = pd.DataFrame({'NTM_Physician_Speciality':list(phys_speciality[2800:])})
add_smoothing_comparison = pd.DataFrame(df['NTM_Physician_Speciality'][0:2800])
add_smoothing_comparison.columns = ['Comp w/ Add. smoothing']

kf = KFold(n_splits=4, shuffle=True, random_state=1111)

# Target encoding for training set using K-folding.
train_te = pd.DataFrame()
for tra_idx, val_idx in kf.split(df_xtrain):
    te = target_encoder.TargetEncoder(handle_missing='return_nan',handle_unknown='return_nan')
    te.fit(df_xtrain.iloc[tra_idx],df_ytrain.iloc[tra_idx])
    temp = te.transform(df_xtrain.iloc[val_idx])
    train_te = pd.concat([train_te,temp],axis=0)

train_te.sort_index(inplace=True)
train_te.columns = ['Speciality_TE w/ K-fold']
train_df_speciality = pd.concat([df_xtrain,train_te,add_smoothing_comparison,df_ytrain],axis=1)

# Target encoding for test set.
te = target_encoder.TargetEncoder(handle_missing='return_nan',handle_unknown='return_nan') #if level is unknown from input data, return nan.
te.fit(df_xtrain,df_ytrain)
test_te = te.transform(df_xtest)
test_te.columns = ['Speciality_TE w/ K-fold']
test_df_speciality = pd.concat([df_xtest,test_te],axis=1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df_1['Race Encoded']= label.fit_transform(df_1.Race)
onehot = OneHotEncoder(categories = 'auto')
onehotx = onehot.fit_transform(df_1['Race Encoded'].values.reshape(-1,1)).toarray()
onehot_race = pd.DataFrame(onehotx)
onehot_race_final = pd.concat([df_1[['Race','Persistency_Flag']], onehot_race], axis =1)
onehot_race_final.columns = ['Race','Persistency','African American','Asian','Caucasian','Hispanic','Other/Uknown']
onehot_race_final.drop(['Race'], axis=1).head(5)

pd.get_dummies(df_1.drop(['Patient_ID'],axis=1)).shape

df_1.groupby('Race')['Persistency_Flag'].value_counts()

df_1['Persistency_Flag'].mean()

race1 = calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=10).head(5)

calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=20).head(5)

calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=200).head(5)

from category_encoders import TargetEncoder
encoder = TargetEncoder()
race_encoded = encoder.fit_transform(df_1['Race'],df_1['Persistency_Flag'])
race_encoded.head(5)

def prescription_risk(col):
    if col == 'HR_VHR':
        return 1
    elif col == 'VLR_LR':
        return -1
    else:
        return None

df['Risk_During_Prescription'] = df['Risk_During_Prescription'].apply(prescription_risk)

def risk_change(col):
    if col == 'No change':
        return 0
    elif col == 'Worsened':
        return -1
    elif col == 'Improved':
        return 1
    else:
        return None

df['Risk_Change'] = df['Risk_Change'].apply(risk_change)

def t_score_during_prescription(col):
    if col == '<=-2.5':
        return -1
    elif col == '>-2.5':
        return 1
    else:
        return None

df['Tscore_During_Prescription'] = df['Tscore_During_Prescription'].apply(t_score_during_prescription)

def change_t_score(col):
    if col == 'No change':
        return 0
    elif col == 'Worsened':
        return -1
    elif col == 'Improved':
        return 1
    else:
        return None

df['Change_T_Score'] = df['Change_T_Score'].apply(change_t_score)

df1 = df.drop(['Tscore_During_Prescription', 'Risk_Change', 'Risk_During_Prescription', 'Patient_ID'], axis=1)

from sklearn.linear_model import LogisticRegression

test_data = df1[df1["Change_T_Score"].isnull()]

y_train = df1["Change_T_Score"]
df1[np.isnan(df['Change_T_Score'])]
y_train = y_train.dropna()
drop_list = list(df1[np.isnan(df['Change_T_Score'])].index)
X_train = df1.drop("Change_T_Score", axis=1)
X_train = X_train.drop(index=drop_list)
X_test = test_data.drop("Change_T_Score", axis=1)

model = LogisticRegression(solver='liblinear', random_state=101)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred2 = list(y_pred)

def null_imputer(column, pred):
    new_col = []
    index = 0
    for value in column:
        if pd.isna(value) == True:
            new_col.append(pred[index])
            print(pred[index])
            index += 1
        else:
            new_col.append(value)
    return new_col

df['Change_T_Score'] = null_imputer(df['Change_T_Score'], y_pred2)

df2 = df.drop(['Tscore_During_Prescription', 'Risk_Change', 'Patient_ID'], axis=1)

test_data = df2[df2["Risk_During_Prescription"].isnull()]

y_train = df2["Risk_During_Prescription"]
y_train = y_train.dropna()
drop_list = list(df2[np.isnan(df['Risk_During_Prescription'])].index)
X_train = df2.drop('Risk_During_Prescription', axis=1)
X_train = X_train.drop(index=drop_list)
X_test = test_data.drop('Risk_During_Prescription', axis=1)

model = LogisticRegression(solver='liblinear', random_state=101)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred3 = list(y_pred)

df['Risk_During_Prescription'] = null_imputer(df['Risk_During_Prescription'], y_pred3)

df3 = df.drop(['Risk_Change', 'Patient_ID'], axis=1)

test_data = df3[df3['Tscore_During_Prescription'].isnull()]

y_train = df3['Tscore_During_Prescription']
y_train = y_train.dropna()
drop_list = list(df3[np.isnan(df['Tscore_During_Prescription'])].index)
X_train = df3.drop('Tscore_During_Prescription', axis=1)
X_train = X_train.drop(index=drop_list)
X_test = test_data.drop('Tscore_During_Prescription', axis=1)

model = LogisticRegression(solver='liblinear', random_state=101)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred4 = list(y_pred)

df['Tscore_During_Prescription'] = null_imputer(df['Tscore_During_Prescription'], y_pred4)

df4 = df.drop(['Patient_ID'], axis=1)

test_data = df4[df4['Risk_Change'].isnull()]

y_train = df4['Risk_Change']
y_train = y_train.dropna()
drop_list = list(df4[np.isnan(df['Risk_Change'])].index)
X_train = df4.drop('Risk_Change', axis=1)
X_train = X_train.drop(index=drop_list)
X_test = test_data.drop('Risk_Change', axis=1)

model = LogisticRegression(solver='liblinear', random_state=101)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred5 = list(y_pred)

df['Risk_Change'] = null_imputer(df['Risk_Change'], y_pred5)

#chi square test (test of independence) ##Livia
#creating a function for chi square test
def chi_square_test(col):
    print('Ho:Persistency is not dependent on '+col)
    print('H1:Persistency is dependent on '+col)
    import scipy.stats as stats #importing stats
    #creating a contigency table
    value_list = df[col].unique().tolist()#creating list of column values
    for value in value_list:
        data_crosstab = pd.crosstab(df['Persistency_Flag']==1,df[col]==value,
                                margins=True, margins_name="Total")
    # significance level
    alpha = 0.05
    # Calcualtion of Chisquare test statistics
    chi_square = 0
    rows = (df['Persistency_Flag']==1).unique()
    columns = (df[col]==value).unique()
    for i in columns:
        for j in rows:
            O = data_crosstab[i][j]
            E = data_crosstab[i]['Total'] * data_crosstab['Total'][j] / data_crosstab['Total']['Total']
            chi_square += (O-E)**2/E
    # The p-value approach
    p_value = 1 - stats.norm.cdf(chi_square, (len(rows)-1)*(len(columns)-1))
    conclusion = "Failed to reject the null hypothesis."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected."

    print("chisquare-score is:", chi_square, " and p value is:", p_value)
    print(conclusion)

#getting chi square test for the categorical columns ##Livia
for col in df:
    print(col)
    chi_square_test(col)
    print(' ')

## Livia
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

##Livia
df_temp = df.drop(['dummy', 'Patient_ID', 'NTM_Physician_Speciality', 'Risk_Count', 'Risk_Change', 'Risk_During_Prescription', 'Tscore_During_Prescription'], axis = 1)
# ^ tentatively dropping
x = df_temp.drop('Persistency_Flag', axis=1) #pulling all independent variables except persistency_flag (dependent variable)

# set dependent variable as y ##Livia
y = df['Persistency_Flag']
y.shape

#test variables ##Livia

test = np.array([1, 0.481340, 0.325028, 19, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
test = test.reshape(1,-1)

x_train, x_test, y_test, y_train = train_test_split(x,y, test_size = 0.3, train_size = 0.7, random_state = 100) ##Livia

## Livia
lm = LogisticRegression(solver='liblinear')
lm.fit(x.values, y.values)

pickle.dump(lm, open('final_model.pkl', 'wb'))
final_model = pickle.load(open('final_model.pkl', 'rb'))

## Livia
predictor = round(final_model.predict(test)[0])

print(predictor)
