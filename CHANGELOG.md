# Changelog

## [1.0.0] - 2025-11-24

[Compare 136d905 ... 5b26d84](https://github.com/Lagrange-Labs/deep-prove-private/compare/136d9055325274d54ba038be114bbc90fa8a69ab...5b26d84cd7eba9167964841f73c7f9f5fd25064f)

### Features



- Added EinSum Layer ([fc4f852](https://github.com/Lagrange-Labs/deep-prove-private/commit/fc4f8523544a666884ce1b6eb74b8933ba2d9a0f))

- Refactor parser logic and add safetensors format for Gemma3 ([0882ab8](https://github.com/Lagrange-Labs/deep-prove-private/commit/0882ab8974018200c97d0b8a3236fd21defaa6a4))

- Store interior mutability ([a8c7a0e](https://github.com/Lagrange-Labs/deep-prove-private/commit/a8c7a0e6beccb548ed7797d6417ea18052a5f96e))

- Chunking claims proving ([234ad53](https://github.com/Lagrange-Labs/deep-prove-private/commit/234ad534f1e3155dd2af7225b22509dd1b480133))

- Introduce tensor handle ([a9f9fc8](https://github.com/Lagrange-Labs/deep-prove-private/commit/a9f9fc80aeeb04ba69003251d3ce75d2012c51bf))

- Introduce a tensor visualization tool ([fbe2fd0](https://github.com/Lagrange-Labs/deep-prove-private/commit/fbe2fd0f2acf032e18d63aa95bea2bbf44f44f66))

- Bench inference alone ([857b2ba](https://github.com/Lagrange-Labs/deep-prove-private/commit/857b2ba9f0f0e29f634bfe4106185c98990802fd))

- Pull models from s3 for worker ([bf36683](https://github.com/Lagrange-Labs/deep-prove-private/commit/bf36683c41eb15dc3ae3004ea7caf829e13cd2c2))

- Handler w/ wrapped tensor ([23ab050](https://github.com/Lagrange-Labs/deep-prove-private/commit/23ab050eab7f198a1069436322962de13c320b17))

- Isolate inference run; add gpt2 ([f8e7759](https://github.com/Lagrange-Labs/deep-prove-private/commit/f8e775987a99fddbed6a52ba76e01e37fb08596e))

- Chunking commitments ([46e6564](https://github.com/Lagrange-Labs/deep-prove-private/commit/46e6564b2eec507190e5c196c6178a6a15bfaf76))

- Replace all linear with einsum ([5d648f6](https://github.com/Lagrange-Labs/deep-prove-private/commit/5d648f6aadc67866542e45ae86b4e6df8a0d0b0f))

- Expose the git hash the library was built from ([9295690](https://github.com/Lagrange-Labs/deep-prove-private/commit/9295690f338f5056bc3de9783f0089a2e49c20f3))


### Bug Fixes



- Einsum wrapped tensor ([eb0b664](https://github.com/Lagrange-Labs/deep-prove-private/commit/eb0b664d00303238cf7fca0e30378c5bc0617d42))

- Update burn to latest ([df95ca9](https://github.com/Lagrange-Labs/deep-prove-private/commit/df95ca9ddeb4cf25c6780a588ecc60c3efcbeb76))

- Update burn to latest ([7656710](https://github.com/Lagrange-Labs/deep-prove-private/commit/7656710e2133d5a2062992fb9ba7176981565b29))

- Remove default value for s3_params_bucket ([dfb2c19](https://github.com/Lagrange-Labs/deep-prove-private/commit/dfb2c19994a633c34089a2c613764fb575e99054))

- Rope outputs shapes with cache ([5b26d84](https://github.com/Lagrange-Labs/deep-prove-private/commit/5b26d84cd7eba9167964841f73c7f9f5fd25064f))


### Refactor



- Convert `DryTensor` to use `KeyTensor` ([62d0675](https://github.com/Lagrange-Labs/deep-prove-private/commit/62d0675a1e93c82d4b2003a8b246f45387280c3f))

- Homogeneize graph traversal ([30db79a](https://github.com/Lagrange-Labs/deep-prove-private/commit/30db79af34b4cbc813965d17168df830a393ccdb))

- Store padded unpadded shapes in tensor ([3f0eff7](https://github.com/Lagrange-Labs/deep-prove-private/commit/3f0eff71c6e81c38d818f4d9fbc1f2910d3b6620))

- Decommission `TensorKey` in favor of `StorageKey` ([c865d94](https://github.com/Lagrange-Labs/deep-prove-private/commit/c865d94064ef7a45f4d51ad0f58e83d8a4f45b7d))

- Flatten trace step structure ([cadfc1d](https://github.com/Lagrange-Labs/deep-prove-private/commit/cadfc1d7e99f5a699fba5632724dd5f4ac229b7a))

- Replace asserts with ensure macros in prod code  ([03eb3a1](https://github.com/Lagrange-Labs/deep-prove-private/commit/03eb3a1cbee442f73fddcd560b53090bc6e16222))


### Documentation



- Rope specs ([69b6611](https://github.com/Lagrange-Labs/deep-prove-private/commit/69b661116b8bee6ecd9241aa9a978a5f313776d9))


### Performance



- Burnify reshape and mha (composite) layer ([796b7db](https://github.com/Lagrange-Labs/deep-prove-private/commit/796b7dbcd5dd4aa62e076ac05f58c6d438b41a28))

- Implement activation int gelu in burn ([39c85bc](https://github.com/Lagrange-Labs/deep-prove-private/commit/39c85bc7dc38d577f658a8b5486fa93177dedf58))

- Add and use pad_next_power_of_two to wrapped tensor ([d2e8cf4](https://github.com/Lagrange-Labs/deep-prove-private/commit/d2e8cf42aa15baea22ebf4dec338b78b21fbb107))


### Testing



- Add some tolerance to einum eval f32 tests ([b463537](https://github.com/Lagrange-Labs/deep-prove-private/commit/b463537319ef6666fb93b6fb25b9809cb50d5898))

- Fix gemma3 test ([d6a37c1](https://github.com/Lagrange-Labs/deep-prove-private/commit/d6a37c180e32d3b08a65f0b2a51f9dc2efa2d4d2))

- Passable tests on CUDA ([597ae63](https://github.com/Lagrange-Labs/deep-prove-private/commit/597ae6337c24f34e053daadcfb4d7ae10d1f571b))

- Improve prove-model benches ([9105db5](https://github.com/Lagrange-Labs/deep-prove-private/commit/9105db5885334d9046440a62715f9d2cb278145d))


### Miscellaneous Tasks



- Remove dead code get_real_weights ([aded4a0](https://github.com/Lagrange-Labs/deep-prove-private/commit/aded4a079c0d7084350eee70b1ed4415b38bccc4))

- Manually run LFS ([e90738a](https://github.com/Lagrange-Labs/deep-prove-private/commit/e90738a3b78efb7e2971acc676de3d7575578a91))

- Remove `~/.gitconfig` if it exists ([a8e9b5f](https://github.com/Lagrange-Labs/deep-prove-private/commit/a8e9b5f5d69b1c72c8b5b2b3315b77bb4c67f99e))

- Gguf cleanup ([61c754f](https://github.com/Lagrange-Labs/deep-prove-private/commit/61c754f1953b0925d97b62ba28fa254ef1809aaa))

- Move conv related elements from tensor.rs ([57353b6](https://github.com/Lagrange-Labs/deep-prove-private/commit/57353b67ba19b06f06ccaec2dd660e304d035b3b))

- Remove mul layer ([8f58734](https://github.com/Lagrange-Labs/deep-prove-private/commit/8f587349be975b7a73cfa719cbe745d3d4f48385))

- Update quantization methods to include output scaling factors and shapes ([c88e314](https://github.com/Lagrange-Labs/deep-prove-private/commit/c88e314f40494c8a7fec2e7853eef852e464e188))

- Bench on nix machines ([5aad242](https://github.com/Lagrange-Labs/deep-prove-private/commit/5aad242d97e3fdfc4703246737d9633413b92c28))

- Add GPU tests ([e7d5046](https://github.com/Lagrange-Labs/deep-prove-private/commit/e7d50468a8b048b729cca2a9043b2912d1b0c21d))

- Improve the run conditions on gpu tests ([b21dfe0](https://github.com/Lagrange-Labs/deep-prove-private/commit/b21dfe0111e7fbc7becdc07b5cca06d67f62d157))

- Add GPU bench ([4f9ce19](https://github.com/Lagrange-Labs/deep-prove-private/commit/4f9ce197405482260f01c0bdb94f618a517737b4))

- Fix the name of gpu testbed on master bench ([f78b329](https://github.com/Lagrange-Labs/deep-prove-private/commit/f78b329212a73b30de64c2fd2f0a6596bbc606f8))

- Bump actions/download-artifact from 4 to 6 ([b9cec01](https://github.com/Lagrange-Labs/deep-prove-private/commit/b9cec01c2a9d5eaa623e36af02dcbc7f285889eb))

- Don't run benches on PR in draft mode ([5f6230f](https://github.com/Lagrange-Labs/deep-prove-private/commit/5f6230f411886599ef8f59530b156d5d76181776))

- Remove proving trace ([8cb3c4b](https://github.com/Lagrange-Labs/deep-prove-private/commit/8cb3c4b199068606edd36e9705d3d416482d0013))


### Security



- Bump actions/upload-artifact from 4 to 5 ([91b40bf](https://github.com/Lagrange-Labs/deep-prove-private/commit/91b40bfb8fac9ac2fac5013b12d3349dce5ab264))

- Bump aquasecurity/trivy-action from 0.33.0 to 0.33.1 ([391b7c6](https://github.com/Lagrange-Labs/deep-prove-private/commit/391b7c60882ba6c11c3ea8a08e2b0276a1b743e3))


### Build



- Push CLI client to Docker Hub as well ([2fb9b39](https://github.com/Lagrange-Labs/deep-prove-private/commit/2fb9b39826c11d5ccca5ee7205a1f45f0d3d64eb))

- Upgrade Ubuntu dockerfiles to 24.04 ([bf8e393](https://github.com/Lagrange-Labs/deep-prove-private/commit/bf8e393ba2bdbd81c6f49b392f24129db32000dc))

- Add trivy action to scan docker image for vulnerabilities ([d5cf528](https://github.com/Lagrange-Labs/deep-prove-private/commit/d5cf52886859bc2ed3f3b73b296f73080e891a42))

- Add feature for CUDA on linux ([02813e9](https://github.com/Lagrange-Labs/deep-prove-private/commit/02813e95c1b068184402ca1ef378b6471508f6c4))

- Update docker pipeline comment push to ECR ([19fe8de](https://github.com/Lagrange-Labs/deep-prove-private/commit/19fe8de4d7909207a1e92fa98cdc41131383686a))

## [0.2.0] - 2025-10-08

### Features

- *(bls)* Impl riscv BLT instruction ([a9dbc95](https://github.com/Lagrange-Labs/deep-prove-private/commit/a9dbc953c61291fb6a5113112630279841ec08dc))

- *(risc_add)* Add blt e2e test ([67a3a65](https://github.com/Lagrange-Labs/deep-prove-private/commit/67a3a6559140cf1a11730c162d3e0f809fc37884))

- *(worker)* Make S3 storage a runtime option ([0e610c8](https://github.com/Lagrange-Labs/deep-prove-private/commit/0e610c8bb989060c22246a2303ec9f969f912d13))



- Type safety ([4a709f5](https://github.com/Lagrange-Labs/deep-prove-private/commit/4a709f5d3d7c8c3f787ea4731a2ce4c317476d37))

- Add counter in ([c6e9010](https://github.com/Lagrange-Labs/deep-prove-private/commit/c6e9010e3ca8314ce59d988a3da4af1227c25985))

- Implement naive version keccak256 circuit ([5e73238](https://github.com/Lagrange-Labs/deep-prove-private/commit/5e73238522536e7699ecf4442f60755652b1dd4a))

- Refactor uint ([5dea054](https://github.com/Lagrange-Labs/deep-prove-private/commit/5dea054faeb6175b3de26232226b4e9e22f9f049))

- Chip handler refactor ([cfd717e](https://github.com/Lagrange-Labs/deep-prove-private/commit/cfd717ee1d0ba9f01e62e4f3e79c0b097303b360))

- Impl riscv BLT instruction ([a9dbc95](https://github.com/Lagrange-Labs/deep-prove-private/commit/a9dbc953c61291fb6a5113112630279841ec08dc))

- Poseidon port ([c011e49](https://github.com/Lagrange-Labs/deep-prove-private/commit/c011e497ebdce568326130c7f6f666de1f9e5bdd))

- Memory chip handler ([5004d94](https://github.com/Lagrange-Labs/deep-prove-private/commit/5004d9486190cc2ab1f14d60a1b58f7613dd9675))

- Add blt e2e test ([67a3a65](https://github.com/Lagrange-Labs/deep-prove-private/commit/67a3a6559140cf1a11730c162d3e0f809fc37884))

- Integrate poseidon port into mpcs + transcript ([a9cd143](https://github.com/Lagrange-Labs/deep-prove-private/commit/a9cd14361c8729a39cc52626d3151598358b0c29))

- Store word ([9d07ff9](https://github.com/Lagrange-Labs/deep-prove-private/commit/9d07ff917a7873b7907249ee40d7170be1acc1d1))

- Support x0 by redirecting writes to RD_NULL ([69d6be7](https://github.com/Lagrange-Labs/deep-prove-private/commit/69d6be7516e86d04dd9c6699c04cc20f0946a5e1))

- Collect all instructions in the VM config ([fb122e8](https://github.com/Lagrange-Labs/deep-prove-private/commit/fb122e8c15093701b94e6381881961f6385797e9))

- Generalize fibonacci example to e2e bin ([24f2474](https://github.com/Lagrange-Labs/deep-prove-private/commit/24f247464588673e7f41bad43ca09077de506662))

- Unconstrained memory init ([17a9d25](https://github.com/Lagrange-Labs/deep-prove-private/commit/17a9d254d1453108603aaa6da06ed6db40a91013))

- Fix padding assignment ([0efb9d1](https://github.com/Lagrange-Labs/deep-prove-private/commit/0efb9d1b380e8eaa49add1fa3a5ca9cecc132c96))

- Log proving khz in e2e.rs ([cec7b82](https://github.com/Lagrange-Labs/deep-prove-private/commit/cec7b82c29f0d06041bf20aec3e3c0932e6643fa))

- Official runner ([bff6707](https://github.com/Lagrange-Labs/deep-prove-private/commit/bff67071575b3a548936775b4f35b0e1cc4eba95))

- Merge blt+bltu circuits ([1d7818e](https://github.com/Lagrange-Labs/deep-prove-private/commit/1d7818ec23c6857ee557532853489ffb23297aa4))

- Matrix multiplication layer ([82b35cd](https://github.com/Lagrange-Labs/deep-prove-private/commit/82b35cdd2130fba709e0d3283693faf9e404aa46))

- Add an elementary DeepProve worker ([c162d38](https://github.com/Lagrange-Labs/deep-prove-private/commit/c162d38f21cd0cf1640e83181eb466c9b70a06df))

- Add env. var. options to the worker ([e186439](https://github.com/Lagrange-Labs/deep-prove-private/commit/e18643930f20b865a3bf2cd14677d1b1f12fdb1c))

- Tensor::pad_to_shape_in_place ([807d4db](https://github.com/Lagrange-Labs/deep-prove-private/commit/807d4dba4fbcfda9f16db5cd8576bae24657e4ff))

- Client authentication ground work ([bd001cc](https://github.com/Lagrange-Labs/deep-prove-private/commit/bd001cc1d220bcf4857d9017c0760264190ab911))

- Minimum memory metrics ([f825323](https://github.com/Lagrange-Labs/deep-prove-private/commit/f8253233729e9df5d27978208d11ae751c5070eb))

- Add memory tracking ([2526764](https://github.com/Lagrange-Labs/deep-prove-private/commit/2526764af421d2168a7a9d779837ea4edcdec8a4))

- Add a command to run the worker locally ([d1def40](https://github.com/Lagrange-Labs/deep-prove-private/commit/d1def40fb88c11985c4ffeace505ea06711d2842))

- Make S3 storage a runtime option ([0e610c8](https://github.com/Lagrange-Labs/deep-prove-private/commit/0e610c8bb989060c22246a2303ec9f969f912d13))

- Add a local-API prover run mode ([0c75b93](https://github.com/Lagrange-Labs/deep-prove-private/commit/0c75b93365d208416566a17bec2d3aee901c1158))

- Integrate AWS marketplace usage metering ([b07bd71](https://github.com/Lagrange-Labs/deep-prove-private/commit/b07bd7193f56d0a076833cd49279edf8276dbda5))

- Add layer proving ([2f62995](https://github.com/Lagrange-Labs/deep-prove-private/commit/2f62995bc9a9dd1d06b6e24b31687845fc3e6141))

- Take ownership of trace when converting to felt ([00f4cde](https://github.com/Lagrange-Labs/deep-prove-private/commit/00f4cde66183ce9a5228dccaf21b515adf0d67bc))

- Argmax proving ([4ccb913](https://github.com/Lagrange-Labs/deep-prove-private/commit/4ccb91314be724c0ec3f2d10f685ae8dc34c63d2))

- Add proving for LayerNorm ([d2190b2](https://github.com/Lagrange-Labs/deep-prove-private/commit/d2190b277b24ab41a8686fd279393c357c2cc776))

- Add temp file cache for S3 data ([8ae76db](https://github.com/Lagrange-Labs/deep-prove-private/commit/8ae76db8c4bde29584e36b02b750058f8bc4b849))

- Util to conver iterator to base field elements ([84f5463](https://github.com/Lagrange-Labs/deep-prove-private/commit/84f5463ea6b059b03c9aca008ee3fa33b2d7fa67))

- Integrate the DP GW-specific HTTP API cli ([0669ba8](https://github.com/Lagrange-Labs/deep-prove-private/commit/0669ba80482c0fcc4a01ad29445ba8c698e0b4f2))

- Splits up ceno into a separate repository ([43c80aa](https://github.com/Lagrange-Labs/deep-prove-private/commit/43c80aa1086008215f5528a5ef8bc4cf57954c7d))

- Implement the local tensor store ([f2ab8e8](https://github.com/Lagrange-Labs/deep-prove-private/commit/f2ab8e81c6ee4cae839c7a2b539197c71160a9fd))

- Return outputs together with proof(s) from requests ([2b89cfe](https://github.com/Lagrange-Labs/deep-prove-private/commit/2b89cfe15843658c7ab4d2b3b56d55127b1273c5))

- Use streaming estimators to gather scaling ranges ([293b02d](https://github.com/Lagrange-Labs/deep-prove-private/commit/293b02d0aba86765a87d4eab4064f24531544aa8))

- Versioning & release automation ([62d9353](https://github.com/Lagrange-Labs/deep-prove-private/commit/62d9353bdf3445dea30301fe66c14582b7a2f983))

- Prove GPT-2 ([0d7bdd7](https://github.com/Lagrange-Labs/deep-prove-private/commit/0d7bdd742410589338d505feda9227ce87f2ca5b))

- Output ancillaries along the proof ([1606e11](https://github.com/Lagrange-Labs/deep-prove-private/commit/1606e116225ce1c096d7189f1f41baeeb3296d68))

- Remove sum-check data from layer context ([ade9ceb](https://github.com/Lagrange-Labs/deep-prove-private/commit/ade9ceb217448c8c47af8a54a4c7f8ded7732003))

- Move to upstream ceno mpcs crate ([a0c1a71](https://github.com/Lagrange-Labs/deep-prove-private/commit/a0c1a7163b2e91eaef5c1be64b9bf11e81a890ce))

- Optimize argmax removing a sum-check ([669d34b](https://github.com/Lagrange-Labs/deep-prove-private/commit/669d34b01b3d5e9dd82bec87f678f565906441f1))

- Graph-based computation for witness commitment generation. ([2015c96](https://github.com/Lagrange-Labs/deep-prove-private/commit/2015c96771d21306217a3afee42d42d2477b6a1c))

- Implement customer authentication ([e6bcd07](https://github.com/Lagrange-Labs/deep-prove-private/commit/e6bcd07ce2a6ecfd28609c4236aa5cb7bc64eb29))

- Enable partition of a graph from coloring ([8a4908b](https://github.com/Lagrange-Labs/deep-prove-private/commit/8a4908bf3faf7a6ae66f185a10c59fe588ac7bc7))

- Homogenize tokenizer logic and add sentencepiece tokenizer (gemma3) ([872ae53](https://github.com/Lagrange-Labs/deep-prove-private/commit/872ae53d851be9a4279fc583008f585a690047d6))

- Reading gemma3 model from GGUF ([320f61c](https://github.com/Lagrange-Labs/deep-prove-private/commit/320f61c93745dd6255d7b959c0a8bee34c0bb4a8))

- Positional rope implementation ([2bebedc](https://github.com/Lagrange-Labs/deep-prove-private/commit/2bebedc0fce79a790fda296230a7893c45d22465))

- Add GLU variant for activation ([4194b42](https://github.com/Lagrange-Labs/deep-prove-private/commit/4194b42961af16015592b111d74fabc5a7a544c2))

- Add the `max_fee` CLI argument ([d540d96](https://github.com/Lagrange-Labs/deep-prove-private/commit/d540d96fb0776b3abf1327b0f93b936bfd45ec04))

- Publish `onnx::from_path` ([57cd56b](https://github.com/Lagrange-Labs/deep-prove-private/commit/57cd56b489a01984733ab765c492791e67d49cde))

- Use global identifiers for static polynomials ([61a981c](https://github.com/Lagrange-Labs/deep-prove-private/commit/61a981c6f652556780896da9a11fa95e729df6be))

- Local attention from gguf ([0bc80bf](https://github.com/Lagrange-Labs/deep-prove-private/commit/0bc80bf122249ad883a0ffabf59e21469b8c9d0e))


### Bug Fixes



- Imm soundness issue ([7e0f194](https://github.com/Lagrange-Labs/deep-prove-private/commit/7e0f19407e029829cd88656412fd540b24090ed4))

- Reliable order of circuits in proofs ([ffd5773](https://github.com/Lagrange-Labs/deep-prove-private/commit/ffd5773b9a748c099c3c572ea746a4f89b9a56d4))

- Use consistent program table size ([a50cfc5](https://github.com/Lagrange-Labs/deep-prove-private/commit/a50cfc565f9ce6301bb04561a760caa190c760af))

- Refactor padding_zero ([c548dc8](https://github.com/Lagrange-Labs/deep-prove-private/commit/c548dc88fccac94fcb529105169ac44b32c66c0c))

- Fix merge conflict ([12ed922](https://github.com/Lagrange-Labs/deep-prove-private/commit/12ed92231220be35574eb176d454be0324825f56))

- Correct table index calculation ([23fbe0a](https://github.com/Lagrange-Labs/deep-prove-private/commit/23fbe0a95ffaa1adc472df3b63d9d315f3f005e9))

- Reshape for flattening is equal to flattening ([3ebc87f](https://github.com/Lagrange-Labs/deep-prove-private/commit/3ebc87fe7a12123eebcae1d0aee57620801ca30f))

- Replace `println!` with logs in non-test codepaths ([3c3da16](https://github.com/Lagrange-Labs/deep-prove-private/commit/3c3da164708bace96d7f06b9898395065890475c))

- Remove parallel iterators to ensure IEEE754 determinism ([bd6431c](https://github.com/Lagrange-Labs/deep-prove-private/commit/bd6431c321668e867d20d60f2943eb47bbfd5554))

- Add more logging to the worker ([fb91761](https://github.com/Lagrange-Labs/deep-prove-private/commit/fb917610d2261a53e3d7b91af1b147083c31687e))

- Test tmp files race ([8f2e053](https://github.com/Lagrange-Labs/deep-prove-private/commit/8f2e053e0f17c05e95f41bd3e1bec8d3a536dfb5))

- Permissions for master branch ci bench ([0254bb0](https://github.com/Lagrange-Labs/deep-prove-private/commit/0254bb04fdeb76535ef3e4aa9b36ef959a17df29))

- Insert returns a new value ([1db85cb](https://github.com/Lagrange-Labs/deep-prove-private/commit/1db85cba0cc2855b941fe75beadbdfb3396e6849))

- Requant intermediate bit size + cache ([12e3f1c](https://github.com/Lagrange-Labs/deep-prove-private/commit/12e3f1cb91e522c4c04a9c830a58be467a3e4e9e))

- Remove another parallel iterator to ensure IEEE754 determinism ([0611452](https://github.com/Lagrange-Labs/deep-prove-private/commit/0611452027622a6f5e45a906126ecf07c55bd642))

- Ci benches paths filter conditions ([2886e5e](https://github.com/Lagrange-Labs/deep-prove-private/commit/2886e5efa614c34f35dc47b80b65b8fe89e81d32))

- Master branch bench wf ([c5bdbb8](https://github.com/Lagrange-Labs/deep-prove-private/commit/c5bdbb8b3035d7ffee7563364116473e126a1081))

- Bench base comparison ([e44ddd4](https://github.com/Lagrange-Labs/deep-prove-private/commit/e44ddd449249d23b76ea884b95b4319eb9a861ab))

- Update to the version with no race in phase1_prove ([4d66da8](https://github.com/Lagrange-Labs/deep-prove-private/commit/4d66da8e82493e21554362ff92c903e828c17315))

- Erroneous check in tenstore ([7e2d2c4](https://github.com/Lagrange-Labs/deep-prove-private/commit/7e2d2c466dae2fc95f3ace87e41845b58a839066))

- Worker name generation ([5a85c4d](https://github.com/Lagrange-Labs/deep-prove-private/commit/5a85c4d91d186646ff27003d19bc2b0020c4a478))

- Fix docker user ([a383b35](https://github.com/Lagrange-Labs/deep-prove-private/commit/a383b3553b828865f89bfab5de2d8b673e473c31))

- Fix and check all features ([3c81a6a](https://github.com/Lagrange-Labs/deep-prove-private/commit/3c81a6a3cda5627b3dd4596586a3249db29b105f))

- Avoid overflow in cubecl kernel max calc ([fc527f9](https://github.com/Lagrange-Labs/deep-prove-private/commit/fc527f9e06d2fbb68e03240befa15c4b973cf340))

- Quantised argmax now only considers unpadded dim size ([0fe292a](https://github.com/Lagrange-Labs/deep-prove-private/commit/0fe292a8b344933344d0bf05fe9466d21d87a42e))

- Update burn and rm bad assert ([6664b37](https://github.com/Lagrange-Labs/deep-prove-private/commit/6664b3702a9be1a77ed24182df7d8c8adf3345f3))

- Remove / downgrade slow tests ([de5f045](https://github.com/Lagrange-Labs/deep-prove-private/commit/de5f045b5a2e29dc874db047aa4649575615e495))

- Add rescaling ([5be9a08](https://github.com/Lagrange-Labs/deep-prove-private/commit/5be9a089f292d1a8227e84c4f9749b3e961d2ab5))


### Refactor



- Uint module ([8ca70a9](https://github.com/Lagrange-Labs/deep-prove-private/commit/8ca70a9a10df1faf4f405979aee0220d53c56143))

- Move circuit-specific logic out of the emulator ([4b2debf](https://github.com/Lagrange-Labs/deep-prove-private/commit/4b2debfeb50ca76b0a2851376052c66dcfbde010))

- Simplify type conversions ([a88b95b](https://github.com/Lagrange-Labs/deep-prove-private/commit/a88b95b95d892eefc2f44a75221bc395307e186b))

- Remove `Option` wrapper around `prog_data` ([e40e2fc](https://github.com/Lagrange-Labs/deep-prove-private/commit/e40e2fc24eaac9b43abe2a49bd641e4e61bf43e0))

- Limit the use of nightly features ([b7b32ca](https://github.com/Lagrange-Labs/deep-prove-private/commit/b7b32ca2033ac2148499f598e50a41f75e0248ab))

- Generalize the use of `Shape` ([a055a23](https://github.com/Lagrange-Labs/deep-prove-private/commit/a055a23ba6c31e9095a21499cdd0b3a891b39f36))

- Split LPN-related components ([2915d19](https://github.com/Lagrange-Labs/deep-prove-private/commit/2915d1954bb68309e0e1b24acf51f7ec8680ad1b))

- Remove exclusive mut context for witness generation ([0721235](https://github.com/Lagrange-Labs/deep-prove-private/commit/0721235a639b6e54e06a81470ba96ce7f93ebbe1))

- Prepare for multi-task GW ([a35250e](https://github.com/Lagrange-Labs/deep-prove-private/commit/a35250e0e82c8eaf0a9c073979aff8accc3df7e2))

- Make NodeId a real type ([cbe8fca](https://github.com/Lagrange-Labs/deep-prove-private/commit/cbe8fca05988fc679f470650ca5663432ae933d9))

- Make `Tensor` opaque ([0291600](https://github.com/Lagrange-Labs/deep-prove-private/commit/0291600c142efe10d6268dfddd14a3369ce83d9d))

- Burnify add layer ([c0d13fa](https://github.com/Lagrange-Labs/deep-prove-private/commit/c0d13fa20cd310aeb0d0443ed69cc13008d1df1a))

- Implement embeddings layer in burn ([655d119](https://github.com/Lagrange-Labs/deep-prove-private/commit/655d119810c05bd55ad1207c14fcf75a8454872a))

- Cache contexts in tests, genericize store ([46644a7](https://github.com/Lagrange-Labs/deep-prove-private/commit/46644a7499cee616a096658107c40ee405ea77d6))

- Improve file cache ergonomics ([e9d1e68](https://github.com/Lagrange-Labs/deep-prove-private/commit/e9d1e6881aef82c6f2bd613d487cc832cfcc22b2))

- Make the graph module more generic ([6120415](https://github.com/Lagrange-Labs/deep-prove-private/commit/61204156f5580c429fc4ff83f053495984eed1de))

- Straighten shapes handling ([5650c43](https://github.com/Lagrange-Labs/deep-prove-private/commit/5650c43950b71061e5b0922d75990c04158e7587))

- Graph logic refactoring, now using port semantics ([4963837](https://github.com/Lagrange-Labs/deep-prove-private/commit/4963837430734ce043a7340ad25251c9e79199a4))


### Documentation



- Document instruction record ([5437316](https://github.com/Lagrange-Labs/deep-prove-private/commit/5437316ce227f48c205b7b6371ea3e54a72fe308))

- Explain the variants of RAM circuits ([289fc59](https://github.com/Lagrange-Labs/deep-prove-private/commit/289fc59bfa6dd6f3e395f7a45e1e4057b68da1b5))

- Rename private input to hints ([a062e13](https://github.com/Lagrange-Labs/deep-prove-private/commit/a062e13ce53ef4fb2ac419541b43f87bf7a6e663))


### Performance



- Use par_chunks to avoid copies ([e78ac2c](https://github.com/Lagrange-Labs/deep-prove-private/commit/e78ac2c1f09f6ae620b854f8ee8a95fe47461970))

- Remove copies ([637d474](https://github.com/Lagrange-Labs/deep-prove-private/commit/637d474221449a81b1ed895e9f5fada051692238))

- Optimise get_perm ([27eb94a](https://github.com/Lagrange-Labs/deep-prove-private/commit/27eb94a60e9d81e23b935de6d44d72bb1100524d))

- Create a criterion benchmark ([55723a5](https://github.com/Lagrange-Labs/deep-prove-private/commit/55723a5322da208eef6486f50e806fac25313130))

- Compute element count during layer witness generation ([13c6941](https://github.com/Lagrange-Labs/deep-prove-private/commit/13c69416caf9b420b1478a9f4c4f022746f3045d))

- Dont clone witness when proving ([64513b1](https://github.com/Lagrange-Labs/deep-prove-private/commit/64513b1c73fb97263a137b4442e093d98809e09b))

- Use GPU acceleration for GeLU ([7b49840](https://github.com/Lagrange-Labs/deep-prove-private/commit/7b49840dec6745d58f2ec4f53f7e1522fc1873fc))

- Optimise the dense layer ([15a229a](https://github.com/Lagrange-Labs/deep-prove-private/commit/15a229ad8115c5a5a4dc8f9840683444c710f90f))

- Implement conv layer for floats in burn ([2272347](https://github.com/Lagrange-Labs/deep-prove-private/commit/227234759035bffa70423cf0c9a66fcf0bb0b014))

- Implement qkv layer in burn ([8016da9](https://github.com/Lagrange-Labs/deep-prove-private/commit/8016da9156425435f4a6e1a1df5e12750a95da65))

- Enable burn fusion with gpu ([693b76f](https://github.com/Lagrange-Labs/deep-prove-private/commit/693b76f8cb30cf14d31dbb4af8af55b771a8a65c))

- Burnify flatten layer ([85b3c36](https://github.com/Lagrange-Labs/deep-prove-private/commit/85b3c36754db5afa5938675961374738091a0906))

- Implement conv layer for element in burn ([70f0558](https://github.com/Lagrange-Labs/deep-prove-private/commit/70f05585a96b8162bd1f0ec783ce52936e4d5729))

- Implement matmul in burn ([6c2117a](https://github.com/Lagrange-Labs/deep-prove-private/commit/6c2117a88ca3b1b9e8402ef87cfaca25fc1162fd))

- Implement logits - argmax in burn ([9503a45](https://github.com/Lagrange-Labs/deep-prove-private/commit/9503a458796fc5faf9b5c6d79a74410a71b3ab55))

- Implement layer norm for i64 in burn ([0e017d3](https://github.com/Lagrange-Labs/deep-prove-private/commit/0e017d3a4fac3ae8e997d97993e404e8711cefe0))

- Implement permute layer in burn ([83471b1](https://github.com/Lagrange-Labs/deep-prove-private/commit/83471b16da6ac7efa8a5ad392cbfdf5aafd6a5a7))

- Implement concat-matmul in burn ([c5f8596](https://github.com/Lagrange-Labs/deep-prove-private/commit/c5f8596820eedbe3811218e30204f9cc3bd6725f))

- Implement burn softmax ([47c0455](https://github.com/Lagrange-Labs/deep-prove-private/commit/47c0455b4bc4cdc4673a8a5cf5fd36418d403780))

- Changed profiling framework to divan ([75ab111](https://github.com/Lagrange-Labs/deep-prove-private/commit/75ab1115dd7238edbd87d9f0a62890f406926ae1))

- Implement requant layer in burn ([bd6be6f](https://github.com/Lagrange-Labs/deep-prove-private/commit/bd6be6f167e463e38c0d1ed25a64886783466c1c))

- Implement pooling layer in burn ([d36e292](https://github.com/Lagrange-Labs/deep-prove-private/commit/d36e292f986af60b414d2ceb6553ab41c3c4fd7f))

- Implement positional absolute in burn ([ad50323](https://github.com/Lagrange-Labs/deep-prove-private/commit/ad5032333aa4f72df77034747534dd6d748bfa56))

- Bench and vectorise tensor ops ([afc3069](https://github.com/Lagrange-Labs/deep-prove-private/commit/afc3069bf7f1dba8732e0a03fa6c77609cf06268))

- Implement rope in burn ([e2b915b](https://github.com/Lagrange-Labs/deep-prove-private/commit/e2b915bed38a3d6eab8367b25c57a9713fd04f53))


### Testing



- Fail if a matching regression test input is no longer present ([097748c](https://github.com/Lagrange-Labs/deep-prove-private/commit/097748c6a786bde962b43222365ab34b427030d8))

- Use seeded rng for tests ([b78f167](https://github.com/Lagrange-Labs/deep-prove-private/commit/b78f16710ce14b86a6181f458a79d9392b168e4a))

- Use proptest strategies to generate tensors for test inputs ([26ea54c](https://github.com/Lagrange-Labs/deep-prove-private/commit/26ea54c7237c6bcc6763b0c6f0fe4bbf0e3af59c))

- Faster tensor proptests ([762e080](https://github.com/Lagrange-Labs/deep-prove-private/commit/762e080baf684c49fbfe216eaa930ae770d5cecb))

- Reduce the max alloc used in concat_matmul benchf ([bdd5f8b](https://github.com/Lagrange-Labs/deep-prove-private/commit/bdd5f8bce872af8bd5b5a77b442b7e5a2338b6e4))


### Miscellaneous Tasks

- *(worker)* Creates a healthcheck endpoint  ([ecf7a87](https://github.com/Lagrange-Labs/deep-prove-private/commit/ecf7a87b7c24861a02510c48aeb9b5c7e59d582d))



- Fix resolver warning ([8890022](https://github.com/Lagrange-Labs/deep-prove-private/commit/8890022bc44284176316e6193cdf622f4b350a1b))

- Make sure all text files end with a newline ([f526d5d](https://github.com/Lagrange-Labs/deep-prove-private/commit/f526d5d4ba6669b8cf0b23954fa2af6d941c30a3))

- Retrieve end cycle & halt code cosmetics ([22fe9b1](https://github.com/Lagrange-Labs/deep-prove-private/commit/22fe9b1b867c3090c08dda48079ae574c45a1a7b))

- Update toolchain ([5f417fb](https://github.com/Lagrange-Labs/deep-prove-private/commit/5f417fb0dc67d1ec131d41e830f0723c4d5491ec))

- Consistent notation ([0539ea0](https://github.com/Lagrange-Labs/deep-prove-private/commit/0539ea0bdd91ab0c12ebe1950918ea81b5a38a1c))

- Copy edit docs ([6ea3bee](https://github.com/Lagrange-Labs/deep-prove-private/commit/6ea3beeceb2b82a51af6d0dfd0bd8e0ef70aabd1))

- Remove `.iter()` ([3a72862](https://github.com/Lagrange-Labs/deep-prove-private/commit/3a72862d39a2a989d028d3bbeea98161b3c1263a))

- Simplify to max_usable_threads ([846033a](https://github.com/Lagrange-Labs/deep-prove-private/commit/846033ad805ae115507ed6923acf37e3ff478279))

- Bump hashbrown to solve Dependabot alerts ([f97ae15](https://github.com/Lagrange-Labs/deep-prove-private/commit/f97ae15438dd18fd6a93ce136081bda326e0d43e))

- Upgrade dependencies, clippy, build profiles ([c2752df](https://github.com/Lagrange-Labs/deep-prove-private/commit/c2752df53be1b4b6832e452b6cbca17e3543556b))

- Formatting, cleanup python files, restore covid ([a879207](https://github.com/Lagrange-Labs/deep-prove-private/commit/a87920729689aafb4a7d33d08c1a98610facf774))

- Add error message ([5e052c1](https://github.com/Lagrange-Labs/deep-prove-private/commit/5e052c1a70dce199f655670f0a3a7dcabe7265b5))

- Typos & formatting ([69c940b](https://github.com/Lagrange-Labs/deep-prove-private/commit/69c940bf5f0125c8806991f042c84aa03b1cf31f))

- Bump python3 msv for zkml ([c6c0ec1](https://github.com/Lagrange-Labs/deep-prove-private/commit/c6c0ec1cfa3c5d3da6995e9e2692da2e28e7c8c7))

- Creates a healthcheck endpoint  ([ecf7a87](https://github.com/Lagrange-Labs/deep-prove-private/commit/ecf7a87b7c24861a02510c48aeb9b5c7e59d582d))

- Jsonify worker logs  ([7eee510](https://github.com/Lagrange-Labs/deep-prove-private/commit/7eee51015fb6fd3e29711c89142d4a6c0d6da60a))

- Remove debug code ([cbbe00d](https://github.com/Lagrange-Labs/deep-prove-private/commit/cbbe00d1970369fbf8f467c108cf4f7d1aba22c2))

- Add Clippy ([4d40181](https://github.com/Lagrange-Labs/deep-prove-private/commit/4d401819aebd82213af074982f14f34aede0169b))

- Resolves merge conflict artifacts  ([9eef464](https://github.com/Lagrange-Labs/deep-prove-private/commit/9eef464e0f3f430db676c6e36aff581e82add6e0))

- Fix the regression test comparison condition ([48527dc](https://github.com/Lagrange-Labs/deep-prove-private/commit/48527dc66257bbffc65c9157b9fe8b65475de3a9))

- Do not run redundant tests ([f3ade9f](https://github.com/Lagrange-Labs/deep-prove-private/commit/f3ade9fac971b4ff2125a7654bc736671d32e8ae))

- Add a PR title linting step ([df62c89](https://github.com/Lagrange-Labs/deep-prove-private/commit/df62c8993348344364e2233395453da790e617da))

- Fix typos, set up `typos` ([dd5d2b0](https://github.com/Lagrange-Labs/deep-prove-private/commit/dd5d2b0fbe9725ab89c8d60becdd3436b82b4042))

- Only run tests when zkML source changed ([a18b8dd](https://github.com/Lagrange-Labs/deep-prove-private/commit/a18b8dd51737f5165a4d6893613f8dc1432f60df))

- Main -> master ([e90c5d7](https://github.com/Lagrange-Labs/deep-prove-private/commit/e90c5d750e9fee32f636623cc26ede6f467fcd92))

- Prune dependencies ([2166699](https://github.com/Lagrange-Labs/deep-prove-private/commit/216669912e51731a06b7d94ce22c38363f931950))

- Paths-filter needs the repo to have been checked out ([d8b2327](https://github.com/Lagrange-Labs/deep-prove-private/commit/d8b2327dd50045bf8544214f8a9ace8415ebc457))

- Add a cache for rust tests ([8bd93a4](https://github.com/Lagrange-Labs/deep-prove-private/commit/8bd93a493a55f52696165659724a5c6bf5c49ac9))

- Add benches for main and PRs ([31d7c4e](https://github.com/Lagrange-Labs/deep-prove-private/commit/31d7c4e1a0ac308ddfd20c50c1ca90a38b081210))

- Fix master branch name for baseline CI bench ([701b313](https://github.com/Lagrange-Labs/deep-prove-private/commit/701b3139e4c30d5504caf0b389c19d5139f62943))

- Fix remove wrong condition on master branch CI bench ([2c1c084](https://github.com/Lagrange-Labs/deep-prove-private/commit/2c1c0843bf933d5fec3634914df7e6a63da4b15b))

- Use percentage threshold for master branch to gather samples ([6575c0b](https://github.com/Lagrange-Labs/deep-prove-private/commit/6575c0b0a974cb2e585727f6510d8bb804d58d78))

- Keep track of RNG seed used in tests ([dbbaf8d](https://github.com/Lagrange-Labs/deep-prove-private/commit/dbbaf8df1f9e4184c16f5887bef3e1490ac124e6))

- Cosmetic changes ([fb6504d](https://github.com/Lagrange-Labs/deep-prove-private/commit/fb6504df633ea60c2d8ccf239052366a83ab0900))

- Fix bench checkout step ([a57c03c](https://github.com/Lagrange-Labs/deep-prove-private/commit/a57c03c22e540359d223946cbffe23eb852ada88))

- Workflows cancel in progress ([b40d487](https://github.com/Lagrange-Labs/deep-prove-private/commit/b40d48779f68c6b557d20fd61207d23f21b666a2))

- Bump ceno fork to use poseidon nocalloc optimisation ([2932bf6](https://github.com/Lagrange-Labs/deep-prove-private/commit/2932bf681ddad198d1e777a63cc80909ee3f03d3))

- Remove unused dependencies ([52cf5a5](https://github.com/Lagrange-Labs/deep-prove-private/commit/52cf5a565205a27315339c82119d9c04be16b200))

- Fix regressions ([011cb8b](https://github.com/Lagrange-Labs/deep-prove-private/commit/011cb8b7f6e442aed4ae981ede9345d3cb37c1c4))

- Bump ceno ([4c99441](https://github.com/Lagrange-Labs/deep-prove-private/commit/4c994419273db0d019d629f918a1a0b603f76920))

- Do not build docker images for draft PRs ([040004d](https://github.com/Lagrange-Labs/deep-prove-private/commit/040004de764a89fbe0b67c606179114b58724b06))

- Use Nix runners for tests & CI ([32328ef](https://github.com/Lagrange-Labs/deep-prove-private/commit/32328ef69b8eb5210ab93713fa533da8edfbd729))

- Fix regressions script call error ([dcb46cc](https://github.com/Lagrange-Labs/deep-prove-private/commit/dcb46cc4de1f250f074f4efe09d5d0364b85354d))

- Add precommit ([bf993d0](https://github.com/Lagrange-Labs/deep-prove-private/commit/bf993d0d6fd89d2de34b77046d363fb75c113627))

- Add typos to pre-commit ([bb0d942](https://github.com/Lagrange-Labs/deep-prove-private/commit/bb0d94250facac415d06405b9004f79e5595045d))

- Do not run quantization tests on draft PRs ([5d3e554](https://github.com/Lagrange-Labs/deep-prove-private/commit/5d3e5547007f28f6180603f3879755f93a11f020))

- Removed school book conv layer ([fd3b32a](https://github.com/Lagrange-Labs/deep-prove-private/commit/fd3b32a1df5f7faf67ac93c475cba7de9184b760))

- Decommission gRPC-based communication ([18a669d](https://github.com/Lagrange-Labs/deep-prove-private/commit/18a669d814bf48a39c101dcfc96fa685b01e4ba8))

- Remove protobuf submodule ([e6183a4](https://github.com/Lagrange-Labs/deep-prove-private/commit/e6183a4948af421a6500ad81f7ce7e69e6a54246))

- Regressions use merge-base commit instead of latest master branch ([ec67717](https://github.com/Lagrange-Labs/deep-prove-private/commit/ec67717644c24fad45c2d22e7d14fcdb4c181c03))

- Cleanup and simplification ([f3838e0](https://github.com/Lagrange-Labs/deep-prove-private/commit/f3838e06c1744fe2c70b25ae31f67cee46d4b107))

- Moved shape to its own mod ([03d8264](https://github.com/Lagrange-Labs/deep-prove-private/commit/03d826425542b16348ec5694150df32a5657ce78))

- Fix master bench action syntax issue ([5c833f0](https://github.com/Lagrange-Labs/deep-prove-private/commit/5c833f002d24371df0e98b93d6a17a9e7cd1c2aa))

- Move to new gkr-backend instead of ceno ([1b2df5b](https://github.com/Lagrange-Labs/deep-prove-private/commit/1b2df5bfff7e4f82f357f9cc5e0e87429a9c56a3))

- More fixes ([1ff4314](https://github.com/Lagrange-Labs/deep-prove-private/commit/1ff4314f57805738e6b870737938c53746eeaa8e))

- Fix pr bench wf ([cdec789](https://github.com/Lagrange-Labs/deep-prove-private/commit/cdec7899a948b1fa7ed1ad2566c7a1c12b9eca10))

- Round to even ([82fdc64](https://github.com/Lagrange-Labs/deep-prove-private/commit/82fdc64b35f1a70dbfb320b633d4d44de77cbab8))

- Fix master branch missing `runs-on` ([696f1f2](https://github.com/Lagrange-Labs/deep-prove-private/commit/696f1f289bbfec4c823732deb1a2431ac52fd658))

- Actions/checkout is broken with LFS ([e0a23f3](https://github.com/Lagrange-Labs/deep-prove-private/commit/e0a23f320916b72be686f27f0963e72b9c97c6aa))

- Fix missing lfs pull for regression's base branch ([b7649d8](https://github.com/Lagrange-Labs/deep-prove-private/commit/b7649d890087439ee3964a219b2bf363600815f4))

- Add clippy/cargo to precommit ([382ae9b](https://github.com/Lagrange-Labs/deep-prove-private/commit/382ae9b8335c22bb1d6eecf60a520bcf2cfecfa6))

- Handle model dir creation ([331ce09](https://github.com/Lagrange-Labs/deep-prove-private/commit/331ce092eb99d99c4bbb2c356afd5d856a847713))

- Fix bench llm arguments ([f93cb25](https://github.com/Lagrange-Labs/deep-prove-private/commit/f93cb25554b658ed50d821d412e12a1e5be0538c))

- Set max 15 samples for bench ([c8ca072](https://github.com/Lagrange-Labs/deep-prove-private/commit/c8ca072ece5f6de0a476504084fbdbf3926c69d2))

- Fix the order of bencher args ([3b9feb1](https://github.com/Lagrange-Labs/deep-prove-private/commit/3b9feb1f41559056551fc8b9ae720d20a3b4f599))

- Fix arg typo ([358427e](https://github.com/Lagrange-Labs/deep-prove-private/commit/358427ee594535dccbdd503683c59f5435c435dc))

- Move number trait to its own mod ([936c274](https://github.com/Lagrange-Labs/deep-prove-private/commit/936c27472e422d473db4b74ce6b4914d3ae5cf2e))

- Remove leftover prost dependency ([bb1d768](https://github.com/Lagrange-Labs/deep-prove-private/commit/bb1d768de1a6d2df68c9642b7cfdd387974e127a))

- Fix the rng seed for proptests in regression tests ([ecefc48](https://github.com/Lagrange-Labs/deep-prove-private/commit/ecefc486d2c49b8aec40f1d5576a97c72ba337b9))

- Update gkr-backend rustc and p3 ([155a95c](https://github.com/Lagrange-Labs/deep-prove-private/commit/155a95c2e51e69a65956d1076c012d335bded6e4))

- Restrict deployment environments to mainnet only ([6897167](https://github.com/Lagrange-Labs/deep-prove-private/commit/68971673bd8ab07f5989a26e9094cff211af1c90))

- Add changelog generation ([d8f18eb](https://github.com/Lagrange-Labs/deep-prove-private/commit/d8f18eb0f5932adf081835fd4c31178b2b8233fa))

- Set initial version ([3906f29](https://github.com/Lagrange-Labs/deep-prove-private/commit/3906f293eb52c0e4e42248842fd790ce4a9adad8))

- `touch` initial CHANGELOG.md ([ecb7fc4](https://github.com/Lagrange-Labs/deep-prove-private/commit/ecb7fc4c02885d006cf22420de681f38a072bff4))

- Release v0.2.0 ([136d905](https://github.com/Lagrange-Labs/deep-prove-private/commit/136d9055325274d54ba038be114bbc90fa8a69ab))


### BaseFold



- Add and reimplement some utility functions. ([1f5b990](https://github.com/Lagrange-Labs/deep-prove-private/commit/1f5b990d934daca4cb1912681419df32fdaadbf2))


### CI



- Pin rust-toolchain to nightly-1.80.0 ([093e930](https://github.com/Lagrange-Labs/deep-prove-private/commit/093e930392eac1ac17c18cadd4bdc4567563efa1))

- More clippy checks and build tool revamps ([21606d9](https://github.com/Lagrange-Labs/deep-prove-private/commit/21606d9ff8c3acaa581a5a085e8f4c74c059865b))


### CICD



- Move on from abandoned `actions-rs/toolchain` ([a773806](https://github.com/Lagrange-Labs/deep-prove-private/commit/a773806af1b17cae111e41f3e886132845f97e38))


### Chore



- Refine RegisterExpr to type ([d03fd2c](https://github.com/Lagrange-Labs/deep-prove-private/commit/d03fd2cc7f1a13b5c18c8300f5a0feeb3e71457a))


### Docs



- Mention the paper ([584ae6c](https://github.com/Lagrange-Labs/deep-prove-private/commit/584ae6c8a2a506d977f7e2c214b5d7d4021b59c7))


### Feat



- Witness should be basefield elements ([41ceaaa](https://github.com/Lagrange-Labs/deep-prove-private/commit/41ceaaab0b329fa92ea18586bd860d2605e04a93))

- Add e2e prover ([0cfa10b](https://github.com/Lagrange-Labs/deep-prove-private/commit/0cfa10bb6e80ea01700bce0db74f82ea8934d935))

- Ecall/Halt ([d3ea040](https://github.com/Lagrange-Labs/deep-prove-private/commit/d3ea0408b5ac075f49d7c7f0b865f9ffc6a9ca1e))

- Implement JAL opcode ([cdb771a](https://github.com/Lagrange-Labs/deep-prove-private/commit/cdb771aba5b41a258f06a776b8d321bc0cf317eb))

- Implement LUI and AUIPC opcodes ([a33748b](https://github.com/Lagrange-Labs/deep-prove-private/commit/a33748b61e2f6ad58ee81a1fb758338944220159))

- Implement JALR opcode ([f6d0aad](https://github.com/Lagrange-Labs/deep-prove-private/commit/f6d0aad2fb6b10f0beeb78499225b796c138381c))

- Add generic impl for store instructions ([0d3f655](https://github.com/Lagrange-Labs/deep-prove-private/commit/0d3f655e384623cdab160469056cc082fa7134a1))

- Add generic impl for load instructions ([1839eb9](https://github.com/Lagrange-Labs/deep-prove-private/commit/1839eb99bfbc1646f3b90d588e2bbf34617ffc10))

- Expose `base_address` and `instructions` decoded from ELF ([045338d](https://github.com/Lagrange-Labs/deep-prove-private/commit/045338d5c3b6a5ecd6bbe2d15b61403bd2327a81))

- Allow constraint_system to take record as input instead of record's rlc ([b472e95](https://github.com/Lagrange-Labs/deep-prove-private/commit/b472e95a849db3a4c1ee2b57070500c33750b2d9))

- Add rw mismatch checkers and lookup checker in mock prover for zkvm ([22df0e4](https://github.com/Lagrange-Labs/deep-prove-private/commit/22df0e4f7d972d380df00c10599c90d71654ab3a))

- Private input integration ([a76d586](https://github.com/Lagrange-Labs/deep-prove-private/commit/a76d586859481af09a69f98bdeeaad4565ef1603))

- Gguf with transformers ([e08996e](https://github.com/Lagrange-Labs/deep-prove-private/commit/e08996ebfa6b7a5a1f3553df4ea446c3e02a9bfd))

- Gguf with transformers ([48b1c56](https://github.com/Lagrange-Labs/deep-prove-private/commit/48b1c563cf8b4f8a0a7d2bd9ea877bba4984433b))

- Llm autoregressive loop ([bec5b93](https://github.com/Lagrange-Labs/deep-prove-private/commit/bec5b9319ac46c5cc3faf1daf50cef01285dedf5))


### Fix



- Avoid running benchmark unit test in the debug model ([2808724](https://github.com/Lagrange-Labs/deep-prove-private/commit/28087246ee9acd1d8bfaa979245ae799eb330c99))

- Imm are not considered as negative in logic i-type instructions ([54d73fc](https://github.com/Lagrange-Labs/deep-prove-private/commit/54d73fc3845501b3954fef8381ee08de64431235))


### Gemm



- Parse alpha and beta and multiply with weight and bias, respectively ([a96784e](https://github.com/Lagrange-Labs/deep-prove-private/commit/a96784e6464512a4afabb2fced4cd1117c52885c))


### MPCS



- Rename CommitmentWithData to CommitmentWithWitness & remove unnecessary trait requirement. ([027f7b0](https://github.com/Lagrange-Labs/deep-prove-private/commit/027f7b0baac71da6443f0cbb8cf557bcbe8e7a00))


### Matmul



- Add proving of bias ([90b3545](https://github.com/Lagrange-Labs/deep-prove-private/commit/90b35452691376e05344bb60e9c9e2a8e8ffb174))


### WIP



- Integration test ([63ad6f9](https://github.com/Lagrange-Labs/deep-prove-private/commit/63ad6f9c35e4d270d8334c71468fcdd21ab454b7))

- Implementing Lookup trait and context ([4097a72](https://github.com/Lagrange-Labs/deep-prove-private/commit/4097a72d928d80dad5f87eaf3068053d1971a53e))

- Implementing Models with Relu ([7d85ba6](https://github.com/Lagrange-Labs/deep-prove-private/commit/7d85ba67da4549ddd59907fc06af9e0c3060aed6))

- Updated lookup protocol ([2448340](https://github.com/Lagrange-Labs/deep-prove-private/commit/24483400a403557b84454daaf1a2edc23ed01418))

- Implementing lookups for requant ([5a16e6d](https://github.com/Lagrange-Labs/deep-prove-private/commit/5a16e6d3f26350b636c67a89d3a6a9223c7b8a08))

- Implemented more of the lookup prover ([3c5dc3c](https://github.com/Lagrange-Labs/deep-prove-private/commit/3c5dc3c28b5b0438cb2725fca3439916f1e35944))

- Compiling lookup witness ([cec070b](https://github.com/Lagrange-Labs/deep-prove-private/commit/cec070b256d7ec9b97d6b2479db930a6c44490dc))

- Requantizing working ([299909d](https://github.com/Lagrange-Labs/deep-prove-private/commit/299909dfa4bcc1eae697fbe9b2aac7987d983f2a))

- Implementing lookups in prover and verifier ([32224a4](https://github.com/Lagrange-Labs/deep-prove-private/commit/32224a4c02d59953a4e3c163e00ddddf9fb5410a))


### [Bug]



- Inconsistency between batch_open and batch_verify ([79e61a1](https://github.com/Lagrange-Labs/deep-prove-private/commit/79e61a1aa85648e4b812bb218c0c44fe3b2623eb))


### Branches



- Refactor BLT to all branches ([b4b0103](https://github.com/Lagrange-Labs/deep-prove-private/commit/b4b010380cef899edd2be41bc17410cac4a38ecf))


### Build

- *(worker)* Do not enable S3 by default ([3bf228d](https://github.com/Lagrange-Labs/deep-prove-private/commit/3bf228dbd6adffa9e21ef554210ee2e841d9be23))



- Do not enable S3 by default ([3bf228d](https://github.com/Lagrange-Labs/deep-prove-private/commit/3bf228dbd6adffa9e21ef554210ee2e841d9be23))

- Use thin LTO for `release`, introduce `fast` with fat LTO ([baf47cf](https://github.com/Lagrange-Labs/deep-prove-private/commit/baf47cfbacca2a926107ec3b0fd96defdb1d3bdc))

- Prepare the worker & client for AWS release ([7d21c35](https://github.com/Lagrange-Labs/deep-prove-private/commit/7d21c35e5e1cb006e413f4a9676333e9e1506a87))

- Add gguf[gui] to python deps ([236f718](https://github.com/Lagrange-Labs/deep-prove-private/commit/236f7184a9de70fdf5c0a04dd9800acaf54b452b))

- Update devenv to 1.8.1 and use rust-overlay ([1eb7cd1](https://github.com/Lagrange-Labs/deep-prove-private/commit/1eb7cd1897298de9e19775c4958cb6179001a82d))

- Upgrade ubuntu in docker builds ([7e4b174](https://github.com/Lagrange-Labs/deep-prove-private/commit/7e4b174bf93355559ae31947c7af0e508665cb78))

- Enable burn metal feat on mac only ([aa4fee8](https://github.com/Lagrange-Labs/deep-prove-private/commit/aa4fee8e19d1ff0cf0495529bad07f222ca23181))

- `protoc` is still a required dependency for ONNX parsing ([2d4b0c2](https://github.com/Lagrange-Labs/deep-prove-private/commit/2d4b0c25dc95ca121c3d84c8dcfdd9e88ce4344d))

- Use vulkan instead of default wgsl ([aef62b1](https://github.com/Lagrange-Labs/deep-prove-private/commit/aef62b1e4a4776a2ef3b76686cb87dc267f067ca))

- Use nextest, only run light tests in draft PRs ([3a0cf01](https://github.com/Lagrange-Labs/deep-prove-private/commit/3a0cf01c64a2b0014e120fb15a9af3324039759f))


### Emul



- Runtime and build scripts ([41aa9f8](https://github.com/Lagrange-Labs/deep-prove-private/commit/41aa9f8751c2bcc9dca63b58f03135f277c726d9))


### Emul-branch



- Move branch logic in its own category ([3a6ddcd](https://github.com/Lagrange-Labs/deep-prove-private/commit/3a6ddcddd5c18a3e17e049b1cffe5bb915ac67d3))


### Emul-decode



- Expose decoded instructions ([a7e13ef](https://github.com/Lagrange-Labs/deep-prove-private/commit/a7e13efda644412921e9c78d9b02fe53ea789393))


### Emul-r0



- Emulator and trace generator ([dae2878](https://github.com/Lagrange-Labs/deep-prove-private/commit/dae2878c213296d7263793bd55e08a4b4c5ea7b1))


### Emul-trace-mem



- Keep track of previous memory op ([58abe5f](https://github.com/Lagrange-Labs/deep-prove-private/commit/58abe5f72e21cd5841af7e17314e06957c717566))


### Emul-visibility



- Reduce visibility of decoder details ([39fd9d3](https://github.com/Lagrange-Labs/deep-prove-private/commit/39fd9d3261c3a4cede9c89be31f0163254c64d41))


### Example-memory



- Use memory in the example ([ccffed5](https://github.com/Lagrange-Labs/deep-prove-private/commit/ccffed5aa47f9fa3d46273de930527e7394980ee))


### Hide-fast-decode



- Rename and privatize specialized instruction form ([534269e](https://github.com/Lagrange-Labs/deep-prove-private/commit/534269ebaf29b5698db65b90c90a5edf3ae087fa))


### Hide-step



- Simplify tests and keep details of StepRecord private ([b300e9b](https://github.com/Lagrange-Labs/deep-prove-private/commit/b300e9bd81f2bb1052ddc8f3e850fc89262df344))


### Limb-bits



- Use named constants ([9355b50](https://github.com/Lagrange-Labs/deep-prove-private/commit/9355b50a3f0b2f60b524df12a3c6d9bbe97fb0c7))


### Mem-addr



- Gadget for memory address alignment ([0da81af](https://github.com/Lagrange-Labs/deep-prove-private/commit/0da81aff2f5589d4b9561ba928b70dd0ee6db7f9))


### Opcode



- Add/sub trace integration and ci pipeline ([71b1aa2](https://github.com/Lagrange-Labs/deep-prove-private/commit/71b1aa291cc0738088fa0822bd7bf15d0fd24d4a))


### Program-data



- Demo how to initialize memory with program data ([fc95429](https://github.com/Lagrange-Labs/deep-prove-private/commit/fc95429cec5be499c36f75a4211dc7636fb67402))


### Refac-mul



- Consolidate MUL with other arithmetic instructions ([ec51621](https://github.com/Lagrange-Labs/deep-prove-private/commit/ec5162170962d1de304ed1dfae3b05447e909147))


### Reg-index



- Expose register index of trace op ([2ca9825](https://github.com/Lagrange-Labs/deep-prove-private/commit/2ca982547edc5578baeeeaa119f1a4e017a93cc9))


### Sync-platform



- Configure the platform / emulator to match circuits ([0cd4234](https://github.com/Lagrange-Labs/deep-prove-private/commit/0cd4234d29c577eaa5edfa1090530fa91f2d5322))


### Wip



- Fixing dense proving ([06cfeee](https://github.com/Lagrange-Labs/deep-prove-private/commit/06cfeeec99cea058763eeeabf3b6c47b45d3d68b))

<!-- generated by git-cliff -->
