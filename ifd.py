from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from PIL import Image,ImageOps
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
data = pd.read_csv('AIS_2017_01_02.csv')
# data = pd.read_csv('datasetmod.csv')
data.isnull().any()
data = data.interpolate(method='linear', axis=0).ffill().bfill()


def run():

   
    #st.sidebar.info('This web app is created for Illegal fishing Detection')
    st.empty()
    
    
    with st.container():
        image_col, text_col = st.columns((1,4))
        with image_col:
            img = Image.open('pngicon.png')
            resizedImg = img.resize((330, 425))
            st.image(resizedImg)

        with text_col:
            st.markdown("<h1 style='text-align: center;'>Illegal Fishing Detection</h1>", unsafe_allow_html=True)
            st.write("""Illegal fishing has become a worldwide concern resulting in drastic ecological consequences due to activities like overfishing. It is statistically shown that about 11-20 million tonnes of fish have been caught illegally on an annual basis, which amounts to 14%-33% of the global annual fishing catch. The estimated illegal fishing catch is totaled to be around $23 Billion. The vessel's ability to dredge, deplete and damage has lowered the fish stock to 65.8% in 2017 from 90% in 1990 within the biologically sustainable levels. To serve the preservation of biodiversity, illegal fishing detection provides an inclusive analysis strategy on the available data from the automatic identification system (AIS), the relative position of a vessel could be identified and the radar detection aids the tracking of vessels. The data is gathered by satellites and terrestrial receivers which is analyzed by The Global Fishing Watch (GFW) organization. The model based on AIS data, speed of the vessel, and vessel type is used to predict the fishing status of a vessel. The model processes the data being fed and targets the vessel by behavior identification and the likelihood of illegal activity could be monitored.""")

            #st.markdown("[Read more...](https://towardsdatascience.com/a-multi-page-interactive-dashboard-with-streamlit-and-plotly-c3182443871a)")
            st.text("")
            st.text("")
            st.text("")

    # image_icon= Image.open('pngicon.png')
    # st.markdown("<h1 style='text-align: center;'>Illegal Fishing Detection</h1>", unsafe_allow_html=True)
    st.subheader("Choose the ship's MMSI number")
    #option = st.text_input('MMSI NUMBER', '367427270')
    option = st.selectbox('',('338124000','367053590','367123040','367533050','371640000','338124000','367176630','367427270','368098000','367004560','373889000','477348200','374553000','356159000','477982600','353003000','255805913','273318720','412436952','311056400','477099600','477135600','355139000','211517000','477027300','636090965','636092722','477108100','355113000','354001000'))
    st.write('The current ship number iS', option)

    

    



    track1 = data[data['MMSI'] == int(option)]    
    track1.plot(kind = 'scatter', x = 'LON', y = 'LAT', figsize=(14,5))


    plt.title("Track of Vessel "+option)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    plt.savefig("fig1.png")
    image_leafy= Image.open('fig1.png')



    numpy.random.seed(7)
    year = pd.DatetimeIndex(track1['BaseDateTime']).year.tolist()
    month = pd.DatetimeIndex(track1['BaseDateTime']).month.tolist()
    day = pd.DatetimeIndex(track1['BaseDateTime']).day.tolist()
    hour = pd.DatetimeIndex(track1['BaseDateTime']).hour.tolist()
    minute = pd.DatetimeIndex(track1['BaseDateTime']).minute.tolist()
    track1['Year'] = year
    track1['month'] = month 
    track1['day'] = day
    track1['hour'] = hour
    track1['minute'] = minute
    track1.corrwith(track1.SOG).sort_values(ascending=False)
    y = track1['SOG'].ravel()
    X = track1[['LAT', 'hour','Cargo', 'COG']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)    
    regressor = SVR(gamma='scale', C = 100000, epsilon=1, degree=3)
    regressor.fit(x_train, y_train)
    yhat = regressor.predict(x_test)
    

    
    

   
    plt.figure(figsize=(14,5))
    labels = ['Obtained value', 'Ideal value']
    plt.plot(y_test, color = 'yellow', label = 'Original Data')
    plt.plot(yhat, color = 'green', label = 'Predicted Data')
    plt.legend(labels)
    plt.show()
    plt.savefig("chart.png")
    showchart= Image.open('chart.png')


   
    yhat = np.round(yhat,1)
    predicted_vs_real = pd.DataFrame({'Actual': y_test,'Predicted': yhat})    
    predicted_vs_real['difference'] = predicted_vs_real['Actual'] - predicted_vs_real['Predicted']
    rounded_difference = predicted_vs_real['difference'].tolist()
    rounded_difference = np.round(rounded_difference,0)
    
    predicted_vs_real['Rounded Difference'] = rounded_difference

   # st.write(predicted_vs_real)
    #st.write("the pred diff is",predicted_vs_real['Rounded Difference'].mean())

    anamolies = predicted_vs_real[(predicted_vs_real['Rounded Difference']>= 6) |(predicted_vs_real['Rounded Difference']<= -6)]

    ##st.write(anamolies)
    ##st.write("The anamoly value of the ship is ",anamolies['Rounded Difference'].mean())

    with st.container():
        st.text("")
        st.write("The anamoly value of the ship is ",anamolies['Rounded Difference'].mean())
        st.text("")
        st.text("")
        st.image(image_leafy)
        st.text("")
        st.text("")
        st.image(showchart)
        st.text("")
        st.text("")
        st.write("This technology strives to categorize the fishing behavior of various vessels and then uses this to target suspicious behaviors in the open ocean. The usage of neural computing aids the crunching of large amounts of data being projected by the sensors and takes care of various discrepancies in the form of noise saving time and resources. The project has integrated geospatially referenced and physics-based sensor data to identify the illicit activity that a vessel partakes in and warns the various agencies that are involved in the protection of the environment and marine biodiversity. Leveraging this technology not only helps in the preservation of the biodiversity of the marine body but also helps the legal fisherman be the beneficiary by eradicating the illegal trade of fishery as much as possible. ")
        st.text("")
        st.text("")
        st.text("")
        st.write("Developed by Vignesh Kumar SM & Mohana Priya S  ")

    # st.text("This application is developed by ")

        # You can call any Streamlit command, including custom components:
        
        
    
        
   
    






















   
    # st.image(image_icon)
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    


def check_password():
    
    
    
    def password_entered():
        
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text("")
        st.text("")
        st.markdown("<h1 style='text-align: center;'>Illegal Fishing Detection</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Enter Username and password to access the vessel details</h4>", unsafe_allow_html=True)
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        with st.container():
            first_col, second_col, third_col = st.columns((1,3,1))
            with first_col:
                st.write("")

            with second_col:
                st.text_input("Username", on_change=password_entered, key="username")
                st.text_input("Password", type="password", on_change=password_entered, key="password")
                st.text("")
                st.text("")
                st.text("")  
                #st.button("Validate User")         
                
            with third_col:
                st.write("")
        return False
            

    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text("")
        st.text("")
        st.markdown("<h1 style='text-align: center;'>Illegal Fishing Detection</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Enter Username and password to access the vessel details</h4>", unsafe_allow_html=True)
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        with st.container():
            first_col, second_col, third_col = st.columns((1,3,1))
            with first_col:
                st.write("")

            with second_col:
                st.text_input("Username", on_change=password_entered, key="username")
                st.text_input("Password", type="password", on_change=password_entered, key="password")
                st.text("")
                st.text("")
                st.text("")
               # st.button("Validate User",on_click=run())
                st.text("")
                st.text("")
                st.error("😕 Uh-oh, check your username and password")
                             
                
            with third_col:
                st.write("")
        return False
        
    else:
        run()
        # Password correct.
        
        
        return True
def check():
    im = Image.open("favicon.ico")
    st. set_page_config( page_title="IFD",
        page_icon=im,layout="wide")

    hide_menu_style = """<style>#MainMenu {visibility: hidden;}</style> """    
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    
    
    
    if check_password():
        st.text("")
        st.text("")
        st.text("")        
        st.empty()
def main():
    check()
    
    





def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def import_predict(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    x=np.amax(prediction)
    return np.argmax(prediction),x 


    
    
    
if __name__ == '__main__':
    main()