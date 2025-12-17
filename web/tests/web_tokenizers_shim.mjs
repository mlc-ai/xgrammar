import { createRequire } from "module";

const require = createRequire(import.meta.url);
const tokenizers = require("@mlc-ai/web-tokenizers/lib/index.cjs");

export const Tokenizer = tokenizers.Tokenizer;
export default tokenizers;
