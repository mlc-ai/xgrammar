# We cannot naively run `yarn test` due to error:
# `TypeError: A dynamic import callback was invoked without --experimental-vm-modules`
# Thus, we need to run `node --experimental-vm-modules node_modules/jest/bin/jest`
# We also need to change to `commonjs` to avoid error `Must use import to load ES Module`.
# Thus we make this edit, run test, and edit back.
# We also need to make this change for web-tokenizers

sed -e s/'"type": "module",'/'"type": "commonjs",'/g -i .backup package.json
sed -e s/'"type": "module",'/'"type": "commonjs",'/g -i .backup node_modules/@mlc-ai/web-tokenizers/package.json
node --experimental-vm-modules node_modules/jest/bin/jest
sed -e s/'"type": "commonjs",'/'"type": "module",'/g -i .backup package.json
sed -e s/'"type": "commonjs",'/'"type": "module",'/g -i .backup node_modules/@mlc-ai/web-tokenizers/package.json

# Cleanup backup files
rm package.json.backup
