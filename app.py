import streamlit as st
from streamlit_player import st_player
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import pickle
import time
import os


file_path = "faiss_store_openai.pkl"
status_text = st.empty()
llm = GoogleGenerativeAI(model = "models/text-bison-001", google_api_key="", temperature=0.5)

def text_to_ai(data, language_code): 
    
    f = open("temp.txt", "w")
    f.write(data)
    f.close()

    loader = TextLoader("temp.txt")
    st.sidebar.text("loading data .......")
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    print("chunks size------------------------",len(data))
       
    #create embeddings and save to FAISS index
    instructor_embedding = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-large")
    embedding = instructor_embedding
    verctorstore_openai = FAISS.from_documents(docs, embedding)
    st.sidebar.text('embedding vector started .......')
    time.sleep(2)

    #save the FAISS index to picke file
    with open(file_path, "wb") as f:
        pickle.dump(verctorstore_openai, f)
        
    st.sidebar.text('embedding vector done .......')

def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try fetching the manual transcript
    try:
        transcript = transcript_list.find_manually_created_transcript()
        language_code = transcript.language_code  # Save the detected language
    except:
        # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
            language_code = transcript.language_code  # Save the detected language
        except:
            # If no auto-generated transcript is found, raise an exception
            raise Exception("No suitable transcript found.")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code  # Return both the transcript and detected language

def summarize_with_langchain(language_code):
    prompt = f'''Summarize the following text in {language_code}.
    
    Add a title to the summary in {language_code}. 
    Include an INTRODUCTION, BULLET POINTS if possible, and a CONCLUSION in {language_code}.'''

    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                verctorstore = pickle.load(f)
                # chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=verctorstore.as_retriever(),input_key ="query", return_source_documents=True)
                chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=verctorstore.as_retriever(),
                        input_key="query",
                        return_source_documents=True
                        )
                result = chain.invoke(prompt)
                # print(result)
                # st.header("Answer")
                # st.subheader(result["result"])
                
                return result
    except Exception as e:
        status_text.write(str(e))
  
def main():
    testVideo = "https://www.youtube.com/watch?v=DBW9aBLBFh4"
    status_text.title('HPE Video Summarizer Demo')
    link = st.sidebar.text_input('Enter video you want to summarize:', testVideo)
    language_code = 0 
    st_player(link)

    if st.sidebar.button('Start Load'):
        if link:
            try:       
                progress = st.progress(0)

                # Embed a youtube video
                # st_player(link)
                
                st.sidebar.text('Loading the transcript...')
                progress.progress(25)

                # Getting both the transcript and language_code
                transcript, language_code = get_transcript(link)
                text_to_ai(transcript, language_code)

                st.sidebar.text(f'AI generated summary, click on Summarize button to View')
                progress.progress(75)

                # status_text.text('Summary:')
                # st.markdown(summary)
                progress.progress(100)
            except Exception as e:
                st.sidebar.write(str(e))
        else:
            st.sidebar.write('Please enter a valid YouTube link.')

    if st.button('Summarize'):
        result = summarize_with_langchain(language_code)
        st.write('Summary:')
        st.write(result["result"])
        

if __name__ == "__main__":
    main()

