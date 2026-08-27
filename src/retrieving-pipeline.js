import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { ChatCerebras } from "@langchain/cerebras";
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import "dotenv/config";


const llm = new ChatCerebras({
    model: "llama3.3-70b",
    temperature: 0,
    maxTokens: undefined,
    maxRetries: 2,
    apiKey: process.env.CEREBRAS_API_KEY
});

//retrieve data from vector DB
export async function queryVectorDB(query) {

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY,
    });

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY,
    });

    const pineconeIndex = pinecone.Index(
        process.env.PINECONE_INDEX_NAME
    );

    const vectorStore = new PineconeStore(embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });

    const result = await vectorStore.similaritySearch(query, 5);
    return result;
}

const question = 'Types of prompt engineering'
const result = await queryVectorDB(question)


export const GENERATE_RESPONSE = PromptTemplate.fromTemplate(
    `
    You are an assistant for question-answering tasks. Use the following pieces of retieved context to answer the queries. If you do not know the answer, just say that you do not know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    `
);


const formatDocumentsAsString = (documents) => {
    return documents.map((doc) => doc?.pageContent).join("\n\n");
};

const docToString = formatDocumentsAsString(result)

const chain = GENERATE_RESPONSE.pipe(llm);

const aiResponse = await chain.invoke({
    question: question,
    context: docToString
});

console.log(aiResponse)