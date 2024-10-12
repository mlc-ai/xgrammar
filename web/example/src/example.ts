import { BNFGrammar, BuiltinGrammar, GrammarStateMatcher, XGTokenTable } from "@mlc-ai/web-xgrammar"
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { Type, Static } from "@sinclair/typebox";

async function getXGTokenTableAndTokenizerFromUrl(
    tokenizerUrl: string,
    decoderType: string
): Promise<[XGTokenTable, Tokenizer]> {
    // 1. Get tokenizer, we use "@mlc-ai/web-tokenizers" here, but any should work
    const jsonBuffer = await (await fetch(tokenizerUrl)).arrayBuffer();
    const tokenizer = await Tokenizer.fromJSON(jsonBuffer);
    // 2. Get raw token table
    const rawTokenTable: string[] = [];
    const vocabSize = tokenizer.getVocabSize();
    for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
        rawTokenTable.push(tokenizer.idToToken(tokenId));
    }
    // 3. Post process token table
    const decodedTokenTable = await XGTokenTable.createXGTokenTable(rawTokenTable, decoderType);
    return [decodedTokenTable, tokenizer];
}

async function jsonExample() {
    console.log("json example");
    const result = await getXGTokenTableAndTokenizerFromUrl(
        "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json",
        "byte_level"
    );
    const decodedTokenTable = result[0];
    const tokenizer = result[1];

    // 1. Initialize grammar state matcher with JSON grammar
    const bnfGrammar: BNFGrammar = await BuiltinGrammar.json();
    const grammarStateMatcher = await GrammarStateMatcher.createGrammarStateMatcher(
        bnfGrammar,
        decodedTokenTable,
    );
    console.log(grammarStateMatcher);

    // 2. Simulated generation of an LLM
    const input = String.raw`{"hi": 1}<|end_of_text|>`;
    const encodedTokens = tokenizer.encode(input);

    // 3. We expect the matcher to accept all tokens generated since it is a valid JSON
    for (let i = 0; i < encodedTokens.length; i++) {
        // 3.1 Generate token bitmask that will modify logits of the LLM
        if (!grammarStateMatcher.isTerminated()) {
            const bitmask = await grammarStateMatcher.findNextTokenBitmask();
            // For debugging, we can check the rejected token IDs from the mask
            const rejectedIDs = await GrammarStateMatcher.getRejectedTokensFromBitmask(
                bitmask,
                grammarStateMatcher.getVocabSize()
            );
        }
        // 3.2 Say the LLM generated `curToken`, which is simulated here, we use `acceptToken()`
        // to update the state of the matcher, so it will generate a new bitmask for the next
        // auto-regressive generation
        const curToken = encodedTokens[i];
        const accepted = grammarStateMatcher.acceptToken(curToken);
        if (!accepted) {
            throw Error("Expect token to be accepted");
        }
    }

    // 4. The last token is and stop token, so the matcher has terminated.
    console.log("grammarStateMatcher.isTerminated(): ", grammarStateMatcher.isTerminated());
    grammarStateMatcher.dispose();
}

async function jsonSchemaExample() {
    console.log("json schema example");
    // 0. Prepare a schema
    const T = Type.Object({
        name: Type.String(),
        house: Type.Enum({
            Gryffindor: "Gryffindor",
            Hufflepuff: "Hufflepuff",
            Ravenclaw: "Ravenclaw",
            Slytherin: "Slytherin",
        }),
        blood_status: Type.Enum({
            "Pure-blood": "Pure-blood",
            "Half-blood": "Half-blood",
            "Muggle-born": "Muggle-born",
        }),
        occupation: Type.Enum({
            Student: "Student",
            Professor: "Professor",
            "Ministry of Magic": "Ministry of Magic",
            Other: "Other",
        }),
        wand: Type.Object({
            wood: Type.String(),
            core: Type.String(),
            length: Type.Number(),
        }),
        alive: Type.Boolean(),
        patronus: Type.String(),
    });

    type T = Static<typeof T>;
    const schema = JSON.stringify(T);
    console.log("schema: ", schema);

    const result = await getXGTokenTableAndTokenizerFromUrl(
        "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json",
        "byte_level"
    );
    const decodedTokenTable = result[0];
    const tokenizer = result[1];

    // 1. Instantiate matcher with a grammar defined by the above schema
    const bnfGrammar: BNFGrammar = await BuiltinGrammar.jsonSchema(schema);
    const grammarStateMatcher = await GrammarStateMatcher.createGrammarStateMatcher(
        bnfGrammar,
        decodedTokenTable,
    );
    console.log(grammarStateMatcher);

    // 2. Simulated generation of an LLM
    const input = String.raw`{
  "name": "Hermione Granger",
  "house": "Ravenclaw",
  "blood_status": "Muggle-born",
  "occupation": "Student",
  "wand": {
    "wood": "Vine",
    "core": "Phoenix Feather",
    "length": 10
  },
  "alive": true,
  "patronus": "Otter"
}<|end_of_text|>`;
    const encodedTokens = tokenizer.encode(input);

    // 3. We expect the matcher to accept all tokens generated since it is a valid JSON
    for (let i = 0; i < encodedTokens.length; i++) {
        // 3.1 Generate token bitmask that will modify logits of the LLM
        if (!grammarStateMatcher.isTerminated()) {
            const bitmask = await grammarStateMatcher.findNextTokenBitmask();
            // For debugging, we can check the rejected token IDs from the mask
            const rejectedIDs = await GrammarStateMatcher.getRejectedTokensFromBitmask(
                bitmask,
                grammarStateMatcher.getVocabSize()
            );
        }
        // 3.2 Say the LLM generated `curToken`, which is simulated here, we use `acceptToken()`
        // to update the state of the matcher, so it will generate a new bitmask for the next
        // auto-regressive generation
        const curToken = encodedTokens[i];
        const accepted = grammarStateMatcher.acceptToken(curToken);
        if (!accepted) {
            throw Error("Expect token to be accepted");
        }
    }

    // 4. The last token is and stop token, so the matcher has terminated.
    console.log("grammarStateMatcher.isTerminated(): ", grammarStateMatcher.isTerminated());
    grammarStateMatcher.dispose();
}

async function testAll() {
    await jsonExample();
    await jsonSchemaExample();
}

testAll();
