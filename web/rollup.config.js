import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import { wasm } from '@rollup/plugin-wasm';
import typescript from 'rollup-plugin-typescript2'

export default {
    input: 'src/index.ts',
    output: [
        {
            file: 'lib/index.js',
            exports: 'named',
            name: 'xgrammar',
            format: 'umd',
            sourcemap: true
        }
    ],
    plugins: [
        nodeResolve({ browser: true }),
        commonjs(),
        wasm(),
        typescript({
            rollupCommonJSResolveHack: false,
            clean: true
        })
    ]
};
