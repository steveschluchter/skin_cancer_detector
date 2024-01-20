import streamlit as st

st.sidebar.success("Paginate on the sidebar, just click!")


linkedin = 'https://www.linkedin.com/in/steven-schluchter-465876b1/'
github = 'https://github.com/steveschluchter/'


st.title("About the app.")
st.title("By: Steve Schluchter")
st.markdown("""
                [Steve on linkedin]('https://www.linkedin.com/in/steven-schluchter-465876b1/')
                
                [Steve on github]('https://www.github.com/steveschluchter/')

                [this app on dockerhub]('https://hub.docker.com/r/sschluch/skinscan')

                [this app github]('https://github.com/steveschluchter/skin_cancer_detector')

                """)

                

st.markdown(""" 
             Steve's skin cancer detector is a streamlit app that ingests a photo and does a deep learning powered assessment of the probability that the images is of cancerous skin.

             Please don't use this as a substitute for actual oncology because it isn't intended to be one.

             The underlying deep learning model was trained on a curated dataset of varied images from the [ISIC Archive](https://api.isic-archive.com/collections/?pinned=true).  
             
             This appp uses uses a neural network that is a combination of the pretrained [VGG19](https://keras.io/api/applications/#usage-examples-for-image-classification-models) and some more architecture added by Steve.

             Thank you for downloading and activating this webapp!

""")

