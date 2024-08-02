import pyaudio, wave, keyboard, time
from langchain_chroma import Chroma
from pydub import AudioSegment
from pydub.playback import play
from langchain_openai.chat_models import ChatOpenAI
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper import load_audio
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv
import librosa
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

#api_key = os.getenv('OPENAI_API_KEY')
api_key = str(input('Please enter your OpenAI api key:'))

client = OpenAI(api_key = api_key)
chat_model = ChatOpenAI(model = 'gpt-4o', api_key=api_key)
vector_db = Chroma(persist_directory = './vector_db', embedding_function = OpenAIEmbeddings(api_key = api_key))
retriever = vector_db.as_retriever(search_kwargs = {'k':10})

parser = StrOutputParser()
template = """
Sistem: Sən Umico.az onlayn alış-veriş platformasının dəstək komandasının bir üzvüsən və insanlara kömək edirsən. Sənin adın Leyladır.
Konteksdən çıxış edərək aşağıdakı suala(suallara) cavab ver. Gündəlik suallara da cavab verə bilərsən. (Məsələn: Salam necəsən? Adın nədir? Sən Kimsən? və s.)
Əgər alış-veriş platforması haqqlndakı suala cavab verə bilmirsənsə, xaiş edirəm "bilmirəm" deyə cavab ver, müştəridən başqa sual istə.

Chat tarixçəsi: {chat_history}

Konteks: {context}

Sual: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | chat_model | parser
processor = WhisperProcessor.from_pretrained("BHOSAI/Pichilti-base-v1")
whisper_model = WhisperForConditionalGeneration.from_pretrained("BHOSAI/Pichilti-base-v1")

chat_history = ""
teacher_template = """
Sistem: Salam sən Azərbaycan dili Müəlliməsisən və yanlış yazılan cümlələri bacardığın qədər düzəldib yenidən yazmalısan, durğu işarələrini, yanlış sözləri hamsını düzəlt.
Yumiko - Umico
Yemək - Umico.
Unutma ki, Sən Umico.az saytının üzvüsən və ola bilər ki şirkətin adını istifadəçi yanlış yazsın. Əgər sözün yazılışı Umico sözünə bənzəyirsə onu Umico olaraq dəyişdir.
Cavab verərkən sadəcə cümlənin düzəlmiş halını yaz, artıq heçbir şey yazma.

Cümlə: {sentence}
"""
teacher_prompt = ChatPromptTemplate.from_template(teacher_template)
teacher_chain = teacher_prompt | chat_model | parser

ai_voice = AudioSegment.from_mp3("greeting.mp3")
play(ai_voice)

time.sleep(3)

while True:

    print("Sizi eşitməyim üçün 'space' düyməsini sıxın.", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    print("Sizi dinləyirəm! Danışığınızı bitirəndə 'space' düyməsini yenidən sıxın.")
    
    audio, frames = pyaudio.PyAudio(), []
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'):
        frames.append(py_stream.read(512))
    py_stream.stop_stream(), py_stream.close(), audio.terminate()
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    
    ai_voice = AudioSegment.from_mp3("waiting.mp3")
    play(ai_voice)
    
    #with open("voice_record.wav", 'rb') as file:
        #buffer_data = file.read()
    
    #payload: FileSource = {
       # "buffer": buffer_data,
    #}

    #options = PrerecordedOptions(
        #model = "nova-2",
        #smart_format=True,
    #)

    #response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    #response = response.to_dict()

    #response = response['results']['channels'][0]['alternatives'][0]['transcript']

    waveform = load_audio('voice_record.wav')
    features = processor(
        waveform, return_tensors = "pt"
    ).input_features
    predicted_ids = whisper_model.generate(features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    response = transcription[0]
    question = teacher_chain.invoke({"sentence": response})
    docs = vector_db.similarity_search(question, k = 5)
    retrieved_docs = [doc.page_content for doc in docs]
    context = "\nSənədlər:\n"
    context += "".join([f"Sənəd {str(i + 1)}:::\n" + doc for i, doc in enumerate(retrieved_docs)])
    answer = chain.invoke({"context": context, "question": question, "chat_history": chat_history})
    chat_history += f"""Müştəri: {question}\n\nLeyla: {answer}"""
    response = client.audio.speech.create(
        model = 'tts-1',
        voice = 'nova',
        input = answer
    )
    response.stream_to_file('output.mp3')
    return_path = 'output.mp3'
    ai_voice = AudioSegment.from_mp3("output.mp3")
    play(ai_voice)