from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

app = Flask(__name__)

# تهيئة مكونات RAG
def initialize_rag():
    # تحميل ملف PDF
    pdf_path = os.path.join('data', 'Data Cleaning -1.pdf')
    pdf_text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    
    # تقسيم النص
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=['\n', '\n\n', ' ', '']
    )
    chunks = text_splitter.split_text(text=pdf_text)
    
    # إنشاء embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # إنشاء vector store
    global vectorstore
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    # إعداد prompt
    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:"""
    global prompt
    prompt = PromptTemplate.from_template(template=prompt_template)

# تهيئة التطبيق
initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']
        
        # إعداد retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        # دالة لتنسيق المستندات
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # إنشاء سلسلة RAG
        cohere_llm = Cohere(
            model="command", 
            temperature=0.1, 
            cohere_api_key='01XCVjqwF2sJPaSB2e2GRCNeGZVVMJiNxJyFpktR'
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | cohere_llm
            | StrOutputParser()
        )
        
        # الحصول على الإجابة
        answer = rag_chain.invoke(question)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)