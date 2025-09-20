from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


load_dotenv()


st.header(" College Research Tool")

l=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta",task='text-generation')

model=ChatHuggingFace(llm=l)

prp_temp=PromptTemplate(
    template=""" Give me Information about "{college_name}"  in "{length_of_passage}" paragraph """,
    input_variables=['college_name','length_of_passage']
)


college_name=st.selectbox("Select College Name",['Galgotias University','Amity University','IIT Bombay','Birla Institute of Technology','Graphic Era University','Manipal University, Jaipur','Banasthali Vidyapith','Christ University'])
length_of_passage=st.selectbox("Length of information",["Short(2-3 lines)","Medium(6-10 lines)","Large(15-20 lines)"])

prompt=prp_temp.format(
    college_name=college_name,
    length_of_passage=length_of_passage
)


if st.button("Summarize"):
    result=model.invoke(prompt)
    st.write(result.content)