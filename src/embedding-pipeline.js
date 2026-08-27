import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import "dotenv/config";

// storing data in Vector DB
export async function docEmbedding(url) {
    const loader = new CheerioWebBaseLoader(url);

    const parseDocs = await loader.load();

    const docsWithMeta = parseDocs.map((doc) => {
        doc.metadata.source = url;
        return doc;
    });

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const allSplits = await textSplitter.splitDocuments(docsWithMeta);

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0", // Native 1024 dimension model
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

    await vectorStore.addDocuments(allSplits);

    console.log("Finished embedding document");
}

await docEmbedding(
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
);