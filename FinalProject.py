#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Section for importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #Livia 


# In[2]:


#df = pd.read_csv('Healthcare_dataset.csv') #Tahsin


# In[3]:


#upload CSV file of dataset
#df = pd.read_csv('/Users/Olivia/Desktop/FinalProject/Healthcare_dataset.csv') #Livia
# Sammy's note - everyone just write your own read csv line so we can each work on the dataset,
# we'll delete it when we're done


# In[4]:


df = pd.read_csv("C:/Users/filto/Desktop/data_glacier/DataGlacierFinalProject/Healthcare_dataset.csv") #Sammy


# In[5]:


#Taking count of rows and columns ## Livia
row_num = len(df.index) #number of rows
print("Number of rows: ", row_num)
col_num = len(df.columns) #number of columns
print("Number of columns: ", col_num)


# In[6]:


#information about the dataframe ## Livia
df.info()


# In[7]:


df.head() ##Livia


# In[8]:


# Replace Column with a comma in the name ## Livia
df.rename(columns = {'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx':'Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx'}, 
          inplace = True)


# In[9]:


# Finding out how many different outputs there are for our data: ## Livia

#Target Variable:
print("Persistency Flag:\n",df.Persistency_Flag.value_counts())
print("\n")


# In[10]:


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


# In[11]:


# NTM ## Livia
print("Physician's Specialty:\n", df.Ntm_Speciality.value_counts())
print("\n")

print("NTM Specialist Flag:\n",df.Ntm_Specialist_Flag.value_counts())
print("\n")

print("NTM Speciality_Bucket:\n",df.Ntm_Speciality_Bucket.value_counts())
print("\n")


# In[12]:


# row 2847
df[df["Ntm_Speciality"] == 'OBSTETRICS & OBSTETRICS & GYNECOLOGY & OBSTETRICS & GYNECOLOGY']


# In[13]:


# row 1291
df[df["Ntm_Speciality"] == 'NEUROLOGY']


# In[14]:


df[df["Ntm_Speciality"] == 'HEMATOLOGY & ONCOLOGY']


# In[15]:


def speciality_converter(col):
    if col == 'ONCOLOGY':
        return 'HEMATOLOGY & ONCOLOGY'
    elif col == 'OBSTETRICS & OBSTETRICS & GYNECOLOGY & OBSTETRICS & GYNECOLOGY':
        return 'OBSTETRICS AND GYNECOLOGY'
    elif col == 'NEUROLOGY':
        return 'PSYCHIATRY AND NEUROLOGY'
    else:
        return col


# In[16]:


df['Ntm_Speciality'] = df['Ntm_Speciality'].map(speciality_converter)

print("Physician's Speciality:\n", df.Ntm_Speciality.value_counts()) ##Livia


# In[17]:


# all information provided by Physician's specialty column, nothing new
df.drop(['Ntm_Speciality_Bucket'], axis=1, inplace=True)


# In[18]:


# Glucocorticoid ## Livia
print("Glucocorticoid Record Prior to NTM:\n",df.Gluco_Record_Prior_Ntm.value_counts())
print("\n")

print("Glucocorticoid Record During to Rx:\n",df.Gluco_Record_During_Rx.value_counts())
print("\n")


# In[19]:


# DEXA Scan?? ## Livia
print("DEXA Scan Frequency During Rx:\n",df.Dexa_Freq_During_Rx.value_counts())
print("\n")

print("DEXA Scan During Rx:\n",df.Dexa_During_Rx.value_counts())
print("\n")


# In[20]:


# all information provided by Dexa freq column
df.drop(['Dexa_During_Rx'], axis=1, inplace=True)


# In[21]:


# Fragility Fracture
print("Fragility Fracture Prior to NTM:\n",df.Frag_Frac_Prior_Ntm.value_counts())
print("\n")

print("Fragility Fracture During Rx:\n",df.Frag_Frac_During_Rx.value_counts())
print("\n")


# In[22]:


# Risk Segment
print("Risk Segment Prior to NTM:\n",df.Risk_Segment_Prior_Ntm.value_counts())
print("\n")

print("Risk Segment During Rx:\n",df.Risk_Segment_During_Rx.value_counts())
print("\n")

print("Change in Risk Segment:\n",df.Change_Risk_Segment.value_counts())
print("\n")


# In[23]:


# T Score ## Livia
print("T-score Bucket Prior to NTM:\n",df.Tscore_Bucket_Prior_Ntm.value_counts())
print("\n")

print("T-score Bucket During Rx:\n",df.Tscore_Bucket_During_Rx.value_counts())
print("\n")

print("Change in T-score:\n",df.Change_T_Score.value_counts())
print("\n")


# In[24]:


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


# In[25]:


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


# In[26]:


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


# In[27]:


#things that could not work in a different section

##Livia

print("Adherence Flag:\n",df.Adherent_Flag.value_counts())
print("\n")


print("Injectable Experience During Rx:\n",df.Injectable_Experience_During_Rx.value_counts())
print("\n")

print("Count of Risks:\n",df.Count_Of_Risks.value_counts())
print("\n")


# In[28]:


df.head()


# In[29]:


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


# In[30]:


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


# In[31]:


#NTM - Nontuberculous mycobacteria
#Gluco - Glucocorticoid
#Dexa - DEXA Scan
#RX - prescription
#Comorb - Comorbidities


# In[32]:


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


# In[33]:


df.head()


# In[34]:


def f(race, ethnicity):
    if ethnicity == 'Hispanic':
        return 'Hispanic'
    else:
        return race


# In[35]:


df['Race'] = df.apply(lambda x: f(x.Race, x.Ethnicity), axis=1)


# In[36]:


df['Race'].head(10)


# In[37]:


df['Race'].value_counts()


# In[38]:


df.drop(['Ethnicity'], axis=1, inplace=True)


# In[39]:


df1 = df.copy()


# In[40]:


df.info()


# In[41]:


#Checking if risk summation match with risk count ##Tahsin
test = []
risks = df.iloc[:,46:65].to_numpy()
for i in risks:
    s = i.sum()
    test.append(s)
    
risk_count = df['Risk_Count']
#print(risks)
np.array_equal(test, risk_count)


# Output 'True' confirms that the summation of the risks for each row is equivalent to the risk count

# In[42]:


'''Giving Age_Bucket ranks for analysis
df.loc[df["Age_Bucket"] == "<55", "Age_Bucket"] = "0"
df.loc[df["Age_Bucket"] == "55-65", "Age_Bucket"] = "1"
df.loc[df["Age_Bucket"] == "65-75", "Age_Bucket"] = "2"
df.loc[df["Age_Bucket"] == ">75", "Age_Bucket"] = "3"
'''


# Sammy's note - $\newline$ I personally disagree with this method of encoding the data as it misinterprets a categorical variable
# as being numeric. Being age 67 would be registered as being 2x the value of someone aged 45. Below, I've put in my own
# way of numerizing the column. It's not perfect as our model might incorrectly perceive someone registered as age 67 as
# different from someone registered as age 72 even though they are both data points in the same category - but it still enables us to use cool ML strategies for numeric data

# Livia's Note
# 
# I get where you're come from with this. depending on what sort of analysis we do though, this is a perfectly valid
# method of analysis in regression. Depending on what we end up doing, we can keep this or drop it.
# 
# Also, I have an idea of how to deal with this whole age issue because we know that age is most definitely a factor
# in this. I'm not sure how yet, but we can use US Census Data to help randomly assign "actual" ages to people within
# an age group.

# In[43]:


#changing case
df.apply(lambda x: x.astype(str).str.lower())


# In[44]:


# not relevant for model training
#df.drop(['Patient_ID'], axis=1, inplace=True)


# In[45]:


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


# In[46]:


races = df['Race']
regions = df['Region']
phys_speciality = df['NTM_Physician_Speciality']
persistence = df['Persistency_Flag']


# In[47]:


phys_speciality


# In[48]:


df['Age'] = df['Age'].apply(age_converter)


# In[49]:


means = df.groupby('Race')['Persistency_Flag'].mean()


# In[50]:


means


# In[51]:


# target encoding on Race column
df['Race'] = df['Race'].map(means)


# In[52]:


len(df['Race'])


# In[53]:


# target encoding on Region column
df['Region'] = df['Region'].map(df.groupby('Region')['Persistency_Flag'].mean())


# In[54]:


df['Region']


# In[55]:


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


# In[56]:


df['Race'] = calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=10)


# In[57]:


df['Region'] = calc_smooth_mean(df, by='Region', on='Persistency_Flag', m=10)


# In[58]:


df['NTM_Physician_Speciality'] = calc_smooth_mean(df, by='Region', on='Persistency_Flag', m=10)


# In[59]:


df['NTM_Physician_Speciality']


# In[60]:


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


# In[61]:


from category_encoders import target_encoder
from sklearn.model_selection import KFold

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


# In[62]:


from category_encoders import target_encoder
from sklearn.model_selection import KFold

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


# In[63]:


train_df_race


# In[64]:


train_df_region


# In[65]:


train_df_speciality


# In[66]:


df.head()


# Sammy's note: Which do you guys thinks works better: smoothing or k-folding? let me know

# In[67]:


df


# Sammy's note: Once we figure out which one we will be applying, this should be it for data preprocessing

# In[68]:


#Tahsin - Testing One Hot Encoding for Race column
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df1['Race Encoded']= label.fit_transform(df1.Race)
onehot = OneHotEncoder(categories = 'auto')
onehotx = onehot.fit_transform(df1['Race Encoded'].values.reshape(-1,1)).toarray()
onehot_race = pd.DataFrame(onehotx)
onehot_race_final = pd.concat([df1[['Race','Persistency_Flag']], onehot_race], axis =1)
onehot_race_final.columns = ['Race','Persistency','African American','Asian','Caucasian','Hispanic','Other/Uknown']
onehot_race_final.drop(['Race'], axis=1).head(5)


# In[69]:


#If One Hot Coding is used, this is how many columns we would have
pd.get_dummies(df).shape


# Tahsin's Note: I wanted to check out One Hot Encoding but the obvious problem is that it creates too many new columns. In fact, the code above shows we would have total 125 columns if we went with this. Hence, I think this should not be used.

# In[70]:


#Label encoding
df1['Race Encoded'].head(5)


# I further wanted to check Label encoding but it makes it seem that one category is greater than the other, when Race is a nominal variable. 

# In[71]:


df1.groupby('Race')['Persistency_Flag'].value_counts()


# In[72]:


df1['Persistency_Flag'].mean()


# In[73]:


race1 = calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=10).head(5)
race1


# In[74]:


calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=20).head(5)


# In[75]:


df['Race']


# Question- would a higher weight (m) value be better?
# 
# Sammy's response: No, I don't think so. Additive smoothing is meant as a response to overfitting when you have a low number of values in your Race column, however, since we have a sizeable enough column length
# (3423 values) its not worth it to overly weigh the global mean

# In[76]:


calc_smooth_mean(df, by='Race', on='Persistency_Flag', m=200).head(5)


# In[77]:


from category_encoders import TargetEncoder
encoder = TargetEncoder()
race_encoded = encoder.fit_transform(df1['Race'],df1['Persistency_Flag'])
race_encoded.head(5)


# In[78]:


df.head()


# In[79]:


df.info()


# In[80]:


df['Risk_During_Prescription'].value_counts()


# In[81]:


def prescription_risk(col):
    if col == 'HR_VHR':
        return 1
    elif col == 'VLR_LR':
        return -1
    else:
        return None


# In[82]:


df['Risk_During_Prescription'] = df['Risk_During_Prescription'].apply(prescription_risk)


# In[83]:


df['Risk_Change'].value_counts()


# In[84]:


def risk_change(col):
    if col == 'No change':
        return 0
    elif col == 'Worsened':
        return -1
    elif col == 'Improved':
        return 1
    else:
        return None


# In[85]:


df['Risk_Change'] = df['Risk_Change'].apply(risk_change)


# In[86]:


df['Tscore_During_Prescription']


# In[87]:


def t_score_during_prescription(col):
    if col == '<=-2.5':
        return -1
    elif col == '>-2.5':
        return 1
    else:
        return None


# In[88]:


df['Tscore_During_Prescription'] = df['Tscore_During_Prescription'].apply(t_score_during_prescription)


# In[89]:


df['Change_T_Score'].value_counts()


# In[90]:


def change_t_score(col):
    if col == 'No change':
        return 0
    elif col == 'Worsened':
        return -1
    elif col == 'Improved':
        return 1
    else:
        return None


# In[91]:


df['Change_T_Score'] = df['Change_T_Score'].apply(change_t_score)


# In[92]:


df[df["Change_T_Score"].isnull()]


# In[93]:


df1 = df.drop(['Tscore_During_Prescription', 'Risk_Change', 'Risk_During_Prescription', 'Patient_ID'], axis=1)
# Livia's Note: I added drop 'Patient_ID' because the following was returning an error

#Tahsin's suggestion: Can you use something other than 'df1' since this was used by me as a copy of df for data viz?


# In[94]:


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


# In[95]:


y_pred2 = list(y_pred)


# In[96]:


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


# In[97]:


df['Change_T_Score'] = null_imputer(df['Change_T_Score'], y_pred2)


# In[98]:


df['Change_T_Score'].value_counts()


# In[99]:


df2 = df.drop(['Tscore_During_Prescription', 'Risk_Change', 'Patient_ID'], axis=1)
#Livia's Note: Same as before; I dropped 'Patient_ID'


# In[100]:


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


# In[101]:


y_pred3 = list(y_pred)


# In[102]:


df['Risk_During_Prescription'] = null_imputer(df['Risk_During_Prescription'], y_pred3)


# In[103]:


df3 = df.drop(['Risk_Change', 'Patient_ID'], axis=1)


# In[104]:


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


# In[105]:


y_pred4 = list(y_pred)


# In[106]:


df['Tscore_During_Prescription'] = null_imputer(df['Tscore_During_Prescription'], y_pred4)


# In[107]:


df4 = df.drop(['Patient_ID'], axis=1)


# In[108]:


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


# In[109]:


y_pred5 = list(y_pred)


# In[110]:


df['Risk_Change'] = null_imputer(df['Risk_Change'], y_pred5)


# In[111]:


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


# In[112]:


#getting chi square test for the categorical columns ##Livia
for col in df:
    print(col)
    chi_square_test(col)
    print(' ')


# # DATA VISUALIZATION

# In[113]:


print("Race:\n",df.Race.value_counts()) ##Livia


# In[114]:


# Race equivalents ## Livia

#0.377264 == Caucasian
#0.331154 == Hispanic
#0.321222 == African American
#0.481340 == Asian
#0.352940 == Other/Unknown


# In[115]:


print("Region:\n", df.Region.value_counts()) ##Livia


# In[116]:


## Region Equivalents ##Livia

#0.325028 == Midwest
#0.395994 == South
#0.442900 == West
#0.420515 == Northeast
#0.410923 == Other/Unknown


# In[117]:


## NTM Physician's Speciality ## Livia
print("Physiscian's Speciality:\n", df.NTM_Physician_Speciality.value_counts())


# In[118]:


train_df_speciality.groupby('NTM_Physician_Speciality')['Comp w/ Add. smoothing'].mean()


# In[119]:


# Persistency Flag ##Livia
from scipy.stats import bernoulli
pval_persistency_flag = (df.Persistency_Flag.value_counts()[1])/df.Persistency_Flag.count()
pval_persistency_flag

bd_persistency = bernoulli(pval_persistency_flag)
x = [0,1]


plt.figure(figsize = (10,10))
plt.xlim(-0.5,1.5)
bar = plt.bar(x, bd_persistency.pmf(x), color = ['lightcoral', 'turquoise'])

plt.bar_label(bar, labels=['No','Yes'])
plt.title("Bernoulli Distribution of the Persistency Flag", fontsize = '20')
#plt.xlabel("Values of Persistency, 0 for 'No', 1 for 'Yes'", fontsize= '15')
plt.ylabel("Probablility", fontsize = '15')


# In[120]:


pval_gender = (df.Gender.value_counts()[1])/df.Gender.count()
pval_gender

bd_gender = bernoulli(pval_gender)
x = [0,1]


plt.figure(figsize = (8,8))
plt.xlim(-0.5,1.5)
bar = plt.bar(x, bd_persistency.pmf(x), color = ['cornflowerblue', 'pink'])

plt.title("Bernoulli Distribution of Gender", fontsize = '20')
plt.bar_label(bar, labels=['Male','Female'])
#plt.xlabel("Values of Gender, 0 for 'Female', 1 for 'Male'", fontsize= '15')
plt.ylabel("Probablility", fontsize = '15')


# In[121]:


df.info()


# In[122]:


cols1 = ['Risk_Type_1_Insulin_Dependent_Diabetes',                                                          
'Risk_Osteogenesis_Imperfecta',                                                                    
'Risk_Rheumatoid_Arthritis',                                                                       
'Risk_Untreated_Chronic_Hyperthyroidism',                                                        
'Risk_Untreated_Chronic_Hypogonadism',                                                             
'Risk_Untreated_Early_Menopause',                                                                  
'Risk_Patient_Parent_Fractured_Their_Hip',                                                         
'Risk_Smoking_Tobacco',                                                                            
'Risk_Chronic_Malnutrition_Or_Malabsorption',                                                      
'Risk_Chronic_Liver_Disease',                                                                      
'Risk_Family_History_Of_Osteoporosis',                                                            
'Risk_Low_Calcium_Intake',                            
'Risk_Vitamin_D_Insufficiency',  
'Risk_Poor_Health_Frailty',                              
'Risk_Excessive_Thinness',              
'Risk_Hysterectomy_Oophorectomy',               
'Risk_Estrogen_Deficiency',                
'Risk_Immobilization',             
'Risk_Recurring_Falls']


# In[123]:


df_melt = pd.melt(df)
df_melt


# In[124]:


risk_prop = []
for name in cols1:
    risk_prop.append(df_melt[df_melt['variable'] == name]['value'].sum()/df_melt[df_melt['variable'] == name]['value'].count())


# In[125]:


prep_df = pd.concat([pd.Series(cols1), pd.Series(risk_prop)], axis=1)


# In[126]:


prep_df.rename(columns = {0:'Risk Type'}, inplace = True)
prep_df.rename(columns = {1:'Risk Proportion'}, inplace = True)


# In[127]:


prep_df.at[0,'Risk Type'] = 'Diabetes'
prep_df.at[1,'Risk Type'] = 'Osteogenesis'
prep_df.at[2,'Risk Type'] = 'Arthritis'
prep_df.at[3,'Risk Type'] = 'Hyperthyroidism'
prep_df.at[4,'Risk Type'] = 'Hypogonadism'
prep_df.at[5,'Risk Type'] = 'Early Menopause'
prep_df.at[6,'Risk Type'] = 'Hip Fracture'
prep_df.at[7,'Risk Type'] = 'Smoking'
prep_df.at[8,'Risk Type'] = 'Malnutrition'
prep_df.at[9,'Risk Type'] = 'Liver Disease'
prep_df.at[10,'Risk Type'] = 'Osteoporosis'
prep_df.at[11,'Risk Type'] = 'Low Calcium'
prep_df.at[12,'Risk Type'] = 'Low Vitamin D'
prep_df.at[13,'Risk Type'] = 'Frailty'
prep_df.at[14,'Risk Type'] = 'Underweight'
prep_df.at[15,'Risk Type'] = 'Hysterectomy'
prep_df.at[16,'Risk Type'] = 'Estrogen'
prep_df.at[17,'Risk Type'] = 'Immobile'
prep_df.at[18,'Risk Type'] = 'Recurring Falls'


# In[128]:


prep_df


# In[129]:


import squarify
random.seed(2)
plt.rcParams['text.color'] = 'black'
plt.figure(figsize=(10,6))
norms = squarify.normalize_sizes(prep_df['Risk Proportion'].to_list(),  dx=3, dy=3)
squarify.plot(sizes=norms, label=prep_df['Risk Type'].to_list(), alpha=1, pad=True)
plt.title('Relative Proportion of Risk Factors Across Patients')
plt.show()


# In[130]:


num_cols = list(df.select_dtypes(['int64']).columns)
def plot_histogram(df, cols, bins=6):
    for col in cols:
        fig = plt.figure(figsize=(8,8))
        ax= fig.gca()
        df[col].plot.hist(ax = ax, bins = bins, color = 'cornflowerblue')
        ax.set_title('Histogram of ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel('Number')
        plt.show()
plot_histogram(df, num_cols)


# In[131]:


df.info()


# In[132]:


bin_cols = list(df.select_dtypes(['uint8']).columns)
def plot_histogram(df, cols, bins=6):
    for col in cols:
        fig = plt.figure(figsize=(8,8))
        ax= fig.gca()
        df[col].plot.hist(ax = ax, bins = bins, color = 'cornflowerblue')
        ax.set_title('Histogram of ' + col)
        ax.set_xlabel(col)
        ax.set_ylabel('Number')
        plt.show()
plot_histogram(df, bin_cols)


# In[133]:


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
plot_catcols(bin_cols, df)


# In[134]:


corr = df.corr()  ##Livia
corr

## Large correlation dataframe, will create smaller correlation heatmaps with persistency flag being common between them.


# In[135]:


# Demographics heatmap ##Livia

df_demo = df[['Persistency_Flag', 'Gender', 'Race', 'Region', 'Age']].copy()
corr_demo = df_demo.corr()

plt.figure(figsize = (16, 9))

# Cutomize the annot
annot_kws={'fontsize':10,                      # To change the size of the font
           'fontstyle':'italic',               # To change the style of font 
           'fontfamily': 'serif',              # To change the family of font 
           'alpha':1 }                         # To change the transparency of the text  


# Customize the cbar
cbar_kws = {"shrink":1,                        # To change the size of the color bar
            'extend':'min',                    # To change the end of the color bar like pointed
            'extendfrac':0.1,                  # To adjust the extension of the color bar
            "drawedges":True,                  # To draw lines (edges) on the color bar
           }
# customize labels

demo_label = ['Persistency Flag', 'Gender', 'Race', 'Region', 'Age']

# take upper correlation matrix
matrix = np.triu(corr_demo)

# Generate heatmap correlation
demo = sns.heatmap(corr_demo, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, 
                 xticklabels = demo_label, yticklabels = demo_label, cbar_kws=cbar_kws)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light')

plt.yticks(
    rotation=30, 
    verticalalignment='center',
    fontweight='light')

# Set the title etc
plt.title('Correlation Heatmap of Demographics with Persistency Flag', fontsize = 20)

# Set the size of text
sns.set(font_scale = 1.2)


# In[136]:


# Risks and Persistency Heatmap sans Total Risk count ##Livia
df_risk = df[['Persistency_Flag', 'Risk_Type_1_Insulin_Dependent_Diabetes', 'Risk_Osteogenesis_Imperfecta', 
             'Risk_Rheumatoid_Arthritis', 'Risk_Untreated_Chronic_Hyperthyroidism', 'Risk_Untreated_Chronic_Hypogonadism',
             'Risk_Untreated_Early_Menopause', 'Risk_Patient_Parent_Fractured_Their_Hip', 'Risk_Smoking_Tobacco',
             'Risk_Chronic_Malnutrition_Or_Malabsorption', 'Risk_Chronic_Liver_Disease', 'Risk_Family_History_Of_Osteoporosis',
             'Risk_Low_Calcium_Intake', 'Risk_Vitamin_D_Insufficiency', 'Risk_Poor_Health_Frailty','Risk_Excessive_Thinness',
             'Risk_Hysterectomy_Oophorectomy', 'Risk_Estrogen_Deficiency', 'Risk_Immobilization', 'Risk_Recurring_Falls']].copy()
corr_risk = df_risk.corr()

plt.figure(figsize = (16, 9))

risk_lab = ['Persistency Flag', 'Type 1 Diabetes', 'Osteogenesis Imperfecta', 'Rheumatoid Arthritis', 
            'Untreated Hyperthyroidism', 'Untreated Hypogonadism','Untreated Early Menopause', 'Fractured Their Hip', 
            'Smoking Tobacco', 'Chronic Malnutrition/Malabsorption', 'Chronic Liver Disease', 
            'Family History of Osteoporosis','Low Calcium Intake', 'Vitamin D Deficiency', 'Poor Health Frailty',
            'Excessive Thinness', 'Hysterectomy Oophorectomy', 'Estrogen Deficiency', 'Immobilization', 
            'Recurring Falls']

# take upper correlation matrix
matrix = np.triu(corr_risk)

# Generate heatmap correlation
ax = sns.heatmap(corr_risk, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, 
                 xticklabels = risk_lab, yticklabels = risk_lab, cbar_kws=cbar_kws)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light')

plt.yticks(
    rotation=0, 
    verticalalignment='center',
    fontweight='light')

# Set the title etc
plt.title('Correlation Heatmap of Risks with Persistency Flag', fontsize = 20)

# Set the size of text
sns.set(font_scale = 1.2)


# In[137]:


# Comorbidities and Persistency Heatmap ##Livia
df_comorb = df[['Persistency_Flag', 'Comorb_Detected_For_Malignant_Neoplasms', 'Comorb_Detected_For_Immunization', 
             'Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx', 'Comorb_Vitamin_D_Deficiency', 
             'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified',
             'Comorb_Encounter_For_Other_Special_Examination_W_O_Complaint_Suspected_or_Reported_Diagnosis', 
             'Comorb_Dorsalgia', 'Comorb_Personal_History_Of_Other_Diseases_And_Conditions',
             'Comorb_Other_Disorders_Of_Bone_Density_And_Structure', 
             'Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias', 
             'Comorb_Osteoporosis_without_current_pathological_fracture',
             'Comorb_Personal_history_of_malignant_neoplasm', 'Comorb_Gastro_esophageal_reflux_disease', 
             'Comorb_Long_Term_Current_Drug_Therapy']].copy()
corr_comorb = df_comorb.corr()

comorb_lab = ['Persistency Flag', 'Detected Malignant Neoplasms', 'Detected for Immunization', 
              'Encounter for General Exam w/o Suspected/Reported Dx', 'Vitamin D Deficiency', 
              'Other Joint Disorder Not Elsewhere Classified',
              'Encounter for Special Exam w/o Complaint (Suspected/Reported Dx)', 'Dorsalgia', 
              'History of Other Diseases/Conditions','Other Bone Density & Structure Disorders', 
              'Lipoprotein Metabolism & Other Lipidemia Disorders', 
              'Osteoporosis w/o Current Pathological Fracture', 'History of Malignant Neoplasm', 
              'Gastro-Esophageal Reflux Disease', 'Long-Term Current Drug Therapy']


plt.figure(figsize = (16, 9))

# take upper correlation matrix
matrix = np.triu(corr_comorb)

# Generate heatmap correlation
ax = sns.heatmap(corr_comorb, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, 
                 xticklabels = comorb_lab, yticklabels = comorb_lab, cbar_kws=cbar_kws)

plt.xticks(
    rotation=35, 
    horizontalalignment='right',
    fontweight='light')

plt.yticks(
    rotation=0, 
    verticalalignment='center',
    fontweight='light')

# Set the title etc
plt.title('Correlation Heatmap of Comorbidities with Persistency Flag', fontsize = 20)

# Set the size of text
sns.set(font_scale = 1.2)


# In[138]:


# Concomitancy and Persistency Heatmap --- ##Livia
## I'm going to replace Concomitancy with 'Treatment Plan' to make it easier to understand
df_concom = df[['Persistency_Flag', 'Concom_Cholesterol_And_Triglyceride_Regulating_Preparations', 
                'Concom_Narcotics', 'Concom_Systemic_Corticosteroids_Plain', 
                'Concom_Anti_Depressants_And_Mood_Stabilisers', 'Concom_Fluoroquinolones','Concom_Cephalosporins', 
                'Concom_Macrolides_And_Similar_Types', 'Concom_Broad_Spectrum_Penicillins', 
                'Concom_Anaesthetics_General', 'Concom_Viral_Vaccines']].copy()
corr_concom = df_concom.corr()

plt.figure(figsize = (16, 9))


concom_lab = ['Persistency Flag', 'Cholesterol & Triglyceride Regulating Preparations', 'Narcotics', 
              'Systemic Corticosteroids Plan', 'Anti-Depressants & Mood Stabilisers', 'Fluoroquinolones',
              'Cephalosporins', 'Macrolides & Similar Types', 'Broad Spectrum Penicillins', 'Anaesthetics: General', 
              'Viral Vaccines']
# take upper correlation matrix
matrix = np.triu(corr_concom)

# Generate heatmap correlation
ax = sns.heatmap(corr_concom, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, 
                 xticklabels = concom_lab, yticklabels = concom_lab, cbar_kws=cbar_kws)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light')

plt.yticks(
    rotation=0, 
    verticalalignment='center',
    fontweight='light')

# Set the title etc
plt.title('Correlation Heatmap of Treatment Plans and Persistency Flag', fontsize = 20)

# Set the size of text
sns.set(font_scale = 1.2)


# In[139]:


# Heatmap of Persistency Flag and NTM Factors ##Livia
df_ntm = df[['Persistency_Flag', 'NTM_Physician_Speciality', 'NTM_Specialist_Flag', 
             'Glucocorticoids_Record_Before_NTM', 'Glucocorticoids_Record_During_prescription', 
             'Fragility_Fracture_Before_NTM','Fragility_Fracture_During_Therapy', 'Risk_Before_NTM', 
             'Tscore_Before_NTM']].copy()
#'Risk_During_Prescription', 'Tscore_During_Prescription', 'Change_T_Score', 'Risk_Change' <-- not applicable
#for heatmaps

corr_ntm = df_ntm.corr()

plt.figure(figsize = (16, 9))

ntm_lab = ['Persistency Flag', 'Physician Speciality', 'Specialist Flag', 'Glucocorticoids Record Before NTM', 
           'Glucocorticoids Record During prescription', 'Fragility Fracture Before NTM',
           'Fragility Fracture During Therapy', 'Risk Before NTM', 'Tscore Before NTM']

## 'Risk During Prescription', 'Tscore During Prescription', 'Change T Score', 'Risk Change'

# take upper correlation matrix
matrix = np.triu(corr_ntm)

# Generate heatmap correlation
ax = sns.heatmap(corr_ntm, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, 
                 xticklabels = ntm_lab, yticklabels = ntm_lab, cbar_kws=cbar_kws)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light')

plt.yticks(
    rotation=0, 
    verticalalignment='center',
    fontweight='light')

# Set the title etc
plt.title('Correlation Heatmap of NTM Factors with Persistency Flag', fontsize = 20)

# Set the size of text
sns.set(font_scale = 1.2)


# In[140]:


# Heatmap of Persistency Flag and Other factors ##Livia
# Am not including 'Race Encoder' as race is already counted above.
df_other = df[['Persistency_Flag', 'DEXA_Scan_Freq_During_Prescription', 'Adherence',
               'Idn_Indicator', 'Injection_Usage_During_Prescription', 'Risk_Count']].copy()
corr_other = df_other.corr()

plt.figure(figsize = (16, 9))

other_lab = ['Persistency Flag', 'DEXA Scan Frequency During Prescription', 'Adherence',
               'IDN Indicator', 'Injection Usage During_Prescription', 'Risk Count']

# take upper correlation matrix
matrix = np.triu(corr_other)

# Generate heatmap correlation
ax = sns.heatmap(corr_other, mask = matrix, cmap = 'rainbow', annot = True, linewidth = 1.5 ,annot_kws= annot_kws, 
                 xticklabels = other_lab, yticklabels = other_lab, cbar_kws=cbar_kws)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light')

plt.yticks(
    rotation=0, 
    verticalalignment='center',
    fontweight='light')

# Set the title etc
plt.title('Correlation Heatmap of Persistency Flag with Other Factors', fontsize = 20)

# Set the size of text
sns.set(font_scale = 1.2)


# In[141]:


import klib #Tahsin
# Livia's Note: Had to do a pip install to get it to run


# In[ ]:


#Age distribution  ##Tahsin
klib.dist_plot(df['Age'])


# In[ ]:


comorb_data = df[['Persistency_Flag','Comorb_Detected_For_Malignant_Neoplasms',
       'Comorb_Detected_For_Immunization',
       'Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx',
       'Comorb_Vitamin_D_Deficiency',
       'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified',
       'Comorb_Encounter_For_Other_Special_Examination_W_O_Complaint_Suspected_or_Reported_Diagnosis',
       'Comorb_Long_Term_Current_Drug_Therapy', 'Comorb_Dorsalgia',
       'Comorb_Personal_History_Of_Other_Diseases_And_Conditions',
       'Comorb_Other_Disorders_Of_Bone_Density_And_Structure',
       'Comorb_Disorders_of_lipoprotein_metabolism_and_other_lipidemias',
       'Comorb_Osteoporosis_without_current_pathological_fracture',
       'Comorb_Personal_history_of_malignant_neoplasm',
       'Comorb_Gastro_esophageal_reflux_disease',]]
concom_data = df[['Persistency_Flag','Concom_Cholesterol_And_Triglyceride_Regulating_Preparations',
       'Concom_Narcotics', 'Concom_Systemic_Corticosteroids_Plain',
       'Concom_Anti_Depressants_And_Mood_Stabilisers',
       'Concom_Fluoroquinolones', 'Concom_Cephalosporins',
       'Concom_Macrolides_And_Similar_Types',
       'Concom_Broad_Spectrum_Penicillins', 'Concom_Anaesthetics_General',
       'Concom_Viral_Vaccines',]]
risk_data = df[['Persistency_Flag','Risk_Type_1_Insulin_Dependent_Diabetes',
       'Risk_Osteogenesis_Imperfecta', 'Risk_Rheumatoid_Arthritis',
       'Risk_Untreated_Chronic_Hyperthyroidism',
       'Risk_Untreated_Chronic_Hypogonadism', 'Risk_Untreated_Early_Menopause',
       'Risk_Patient_Parent_Fractured_Their_Hip', 'Risk_Smoking_Tobacco',
       'Risk_Chronic_Malnutrition_Or_Malabsorption',
       'Risk_Chronic_Liver_Disease', 'Risk_Family_History_Of_Osteoporosis',
       'Risk_Low_Calcium_Intake', 'Risk_Vitamin_D_Insufficiency',
       'Risk_Poor_Health_Frailty', 'Risk_Excessive_Thinness',
       'Risk_Hysterectomy_Oophorectomy', 'Risk_Estrogen_Deficiency',
       'Risk_Immobilization', 'Risk_Recurring_Falls', 'Risk_Count']]


# In[ ]:


#A cleaner look at the correlation of features against Persistency, ranked ##Tahsin
ax= klib.corr_plot(comorb_data, target= 'Persistency_Flag')
ax.title.set_text('Comorbidities and Persistency Correlation ranked ')


# In[ ]:


ax1 = klib.corr_plot(concom_data, target= 'Persistency_Flag') ##Tahsin
ax1.title.set_text('Concomitancy and Persistency Correlation ranked ')


# In[ ]:


ax2 = klib.corr_plot(risk_data, target= 'Persistency_Flag')
ax2.title.set_text('Risk and Persistency Correlation ranked (Risk Count included)')


# In[ ]:


#Boxplot shows outliers in risk count ##Tahsin
sns.boxplot(data=df1['Risk_Count'])


# In[ ]:


df1['Region'].value_counts()


# In[ ]:


fig, ax = plt.subplots()
ax.pie(df1['Race'].value_counts(),autopct='%1.1f%%',labels = ['Caucasian','Hispanic','African American','Asian','Other'],
        shadow=True)
plt.title('Pie Chart of Race')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.pie(df1['Gender'].value_counts(),autopct='%1.1f%%',explode = (0, 0.1),labels = ['Female','Male'],
        shadow=True)
plt.title('Pie Chart of Gender')
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.pie(df1['Region'].value_counts(),autopct='%1.1f%%',labels = ['Midwest','South','West','Northeast','Other'],
        shadow=True)
plt.title('Pie Chart of Region')
plt.show()


# In[ ]:





# In[ ]:


# plt.ylim(0, 0.06)
# df[df['Risk_Count'] >= 4]


# In[ ]:


Risk5 = df[df['Risk_Count'] >= 5]['Patient_ID']


# In[ ]:


print(Risk5)


# In[ ]:


plt.figure(figsize=(9,6))
plt.legend(prop={'size':10})
plt.ylim(4, 8)
plt.xticks(rotation = 75)
fig = sns.barplot(data=df[df['Risk_Count'] >= 5], x='Patient_ID', y='Risk_Count', hue='Persistency_Flag')
    
plt.ylabel("Risk Count", fontsize = 15) ##Livia
plt.xlabel("Patient ID's with more than 4 Risks", fontsize = 15) ##Livia
plt.title("Persistency of Patients with more than 4 Risks", fontsize = 20) ##Livia


# In[ ]:


# Livia's Population Visual
plt.figure(figsize=(10,4))
sns.histplot(x=df['Age'],bins=50,kde=True, color = 'cornflowerblue')
plt.axvline(x=df.Age.mean(), color='red', label = 'Mean: 73.6 years old')
plt.axvline(x=df.Age.median(), color='blue', ls='--', lw=2.5, label = 'Median: 73 years old')
plt.legend(loc="upper left", fontsize = '13')
plt.title("Graph of Patient Ages", fontsize = '20')
plt.tight_layout()
plt.show()


# # Data Modelling:

# In[183]:


## Livia
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


# In[184]:


# Beginning data modelling process ##Livia
df.info()


# In[185]:


df.shape #get shape in order to figure out what to pull ##Livia


# In[186]:


##Livia
df_temp = df.drop(['dummy', 'Patient_ID', 'NTM_Physician_Speciality', 'Risk_Count', 'Risk_Change', 'Risk_During_Prescription', 'Tscore_During_Prescription'], axis = 1)
# ^ tentatively dropping 
x = df_temp.drop('Persistency_Flag', axis=1) #pulling all independent variables except persistency_flag (dependent variable)


# In[187]:


x.shape #confirming that it is the right shape ##Livia


# In[188]:


x.info() #confirming the contents are only independent variables ##Livia


# In[189]:


# set dependent variable as y ##Livia
y = df['Persistency_Flag'] 
y.shape


# In[190]:


y.info() ## Livia


# In[191]:


y ##Livia


# In[192]:


#getting info for testing ## Livia

pd.set_option('display.max_columns', None)
print(x.loc[[1]]) #using first row to build my 'test'


# In[201]:


#test variables ##Livia

test = np.array([1, 0.481340, 0.325028, 19, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
test = test.reshape(1,-1)


# In[202]:


test.shape #Livia


# In[203]:


x_train, x_test, y_test, y_train = train_test_split(x,y, test_size = 0.3, train_size = 0.7, random_state = 100) ##Livia


# In[208]:


## Livia
lm = LogisticRegression(solver='liblinear')
lm.fit(x.values, y.values)


# In[211]:


##Livia

pickle.dump(lm, open('final_model.pkl', 'wb'))
final_model = pickle.load(open('final_model.pkl', 'rb'))


# In[210]:


## Livia
predictor = round(final_model.predict(test)[0])

print(predictor)


# In[181]:


#Flask deployment in seperate file called 'app_final.py'
##Livia


# In[ ]:





# In[ ]:




