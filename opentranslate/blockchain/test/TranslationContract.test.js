const { expect } = require("chai");
const { ethers } = require("hardhat");
const config = require("../config/contracts");

describe("TranslationContract", function () {
    let token;
    let translation;
    let owner;
    let translator;
    let validator1;
    let validator2;
    let addrs;

    beforeEach(async function () {
        // Get signers
        [owner, translator, validator1, validator2, ...addrs] = await ethers.getSigners();

        // Deploy token
        const PUMPFUNToken = await ethers.getContractFactory("PUMPFUNToken");
        token = await PUMPFUNToken.deploy();
        await token.deployed();

        // Deploy translation contract
        const TranslationContract = await ethers.getContractFactory("TranslationContract");
        translation = await TranslationContract.deploy(token.address);
        await translation.deployed();

        // Initialize translation contract
        await translation.initialize(
            ethers.utils.parseEther(config.translation.minimumStake),
            config.translation.minimumValidations,
            config.translation.minimumScore,
            ethers.utils.parseEther(config.translation.baseReward),
            config.translation.goodScoreMultiplier,
            config.translation.poorScoreMultiplier,
            config.translation.validatorRewardMultiplier
        );

        // Setup test accounts
        const stakeAmount = ethers.utils.parseEther(config.translation.minimumStake);
        
        // Transfer and stake tokens for translator
        await token.transfer(translator.address, stakeAmount * 2);
        await token.connect(translator).stake(stakeAmount);

        // Transfer and stake tokens for validators
        await token.transfer(validator1.address, stakeAmount * 2);
        await token.connect(validator1).stake(stakeAmount);
        await token.transfer(validator2.address, stakeAmount * 2);
        await token.connect(validator2).stake(stakeAmount);

        // Transfer tokens to contract for rewards
        await token.transfer(
            translation.address,
            ethers.utils.parseEther(config.translation.rewards.contractAmount)
        );
    });

    describe("Initialization", function () {
        it("Should set the correct token address", async function () {
            expect(await translation.token()).to.equal(token.address);
        });

        it("Should set the correct minimum stake", async function () {
            expect(await translation.minimumStake()).to.equal(
                ethers.utils.parseEther(config.translation.minimumStake)
            );
        });

        it("Should set the correct minimum validations", async function () {
            expect(await translation.minimumValidations()).to.equal(
                config.translation.minimumValidations
            );
        });

        it("Should set the correct minimum score", async function () {
            expect(await translation.minimumScore()).to.equal(
                config.translation.minimumScore
            );
        });

        it("Should fail if initialized twice", async function () {
            await expect(
                translation.initialize(
                    ethers.utils.parseEther(config.translation.minimumStake),
                    config.translation.minimumValidations,
                    config.translation.minimumScore,
                    ethers.utils.parseEther(config.translation.baseReward),
                    config.translation.goodScoreMultiplier,
                    config.translation.poorScoreMultiplier,
                    config.translation.validatorRewardMultiplier
                )
            ).to.be.revertedWith("Already initialized");
        });
    });

    describe("Translation Submission", function () {
        const sourceText = "Hello, world!";
        const targetText = "¡Hola, mundo!";
        const sourceLang = "en";
        const targetLang = "es";
        const domain = "general";

        it("Should allow translator to submit translation", async function () {
            const tx = await translation
                .connect(translator)
                .submitTranslation(
                    sourceText,
                    targetText,
                    sourceLang,
                    targetLang,
                    domain
                );
            
            const receipt = await tx.wait();
            expect(receipt.events[0].event).to.equal("TranslationSubmitted");
            
            const translationId = await translation.getTranslationCount();
            const translationData = await translation.getTranslation(translationId);
            
            expect(translationData.translator).to.equal(translator.address);
            expect(translationData.sourceText).to.equal(sourceText);
            expect(translationData.targetText).to.equal(targetText);
            expect(translationData.sourceLang).to.equal(sourceLang);
            expect(translationData.targetLang).to.equal(targetLang);
            expect(translationData.domain).to.equal(domain);
            expect(translationData.status).to.equal(0); // Pending
        });

        it("Should fail if translator has insufficient stake", async function () {
            // Unstake tokens
            await token.connect(translator).unstake(
                ethers.utils.parseEther(config.translation.minimumStake)
            );

            await expect(
                translation
                    .connect(translator)
                    .submitTranslation(
                        sourceText,
                        targetText,
                        sourceLang,
                        targetLang,
                        domain
                    )
            ).to.be.revertedWith("Insufficient stake");
        });
    });

    describe("Translation Validation", function () {
        let translationId;

        beforeEach(async function () {
            // Submit a translation
            const tx = await translation
                .connect(translator)
                .submitTranslation(
                    "Hello, world!",
                    "¡Hola, mundo!",
                    "en",
                    "es",
                    "general"
                );
            
            const receipt = await tx.wait();
            translationId = await translation.getTranslationCount();
        });

        it("Should allow validator to validate translation", async function () {
            const score = 85;
            const feedback = "Good translation, but could be more natural";

            const tx = await translation
                .connect(validator1)
                .validateTranslation(translationId, score, feedback);
            
            const receipt = await tx.wait();
            expect(receipt.events[0].event).to.equal("TranslationValidated");

            const validationCount = await translation.getValidationCount(translationId);
            expect(validationCount).to.equal(1);

            const validation = await translation.getValidation(translationId, 0);
            expect(validation.validator).to.equal(validator1.address);
            expect(validation.score).to.equal(score);
            expect(validation.feedback).to.equal(feedback);
        });

        it("Should fail if validator has insufficient stake", async function () {
            // Unstake tokens
            await token.connect(validator1).unstake(
                ethers.utils.parseEther(config.translation.minimumStake)
            );

            await expect(
                translation
                    .connect(validator1)
                    .validateTranslation(translationId, 85, "Good translation")
            ).to.be.revertedWith("Insufficient stake");
        });

        it("Should fail if validator validates same translation twice", async function () {
            await translation
                .connect(validator1)
                .validateTranslation(translationId, 85, "Good translation");

            await expect(
                translation
                    .connect(validator1)
                    .validateTranslation(translationId, 90, "Very good translation")
            ).to.be.revertedWith("Already validated by this validator");
        });
    });

    describe("Translation Completion", function () {
        let translationId;

        beforeEach(async function () {
            // Submit a translation
            const tx = await translation
                .connect(translator)
                .submitTranslation(
                    "Hello, world!",
                    "¡Hola, mundo!",
                    "en",
                    "es",
                    "general"
                );
            
            const receipt = await tx.wait();
            translationId = await translation.getTranslationCount();

            // Add validations
            await translation
                .connect(validator1)
                .validateTranslation(translationId, 85, "Good translation");
            await translation
                .connect(validator2)
                .validateTranslation(translationId, 90, "Very good translation");
        });

        it("Should complete translation with good score", async function () {
            const tx = await translation
                .connect(validator1)
                .validateTranslation(translationId, 95, "Excellent translation");
            
            const receipt = await tx.wait();
            expect(receipt.events[0].event).to.equal("TranslationCompleted");

            const translationData = await translation.getTranslation(translationId);
            expect(translationData.status).to.equal(1); // Completed
            expect(translationData.averageScore).to.equal(90); // (85 + 90 + 95) / 3

            // Check rewards
            const translatorScore = await translation.getTranslatorScore(translator.address);
            expect(translatorScore).to.be.gt(0);
        });

        it("Should fail translation with poor score", async function () {
            const tx = await translation
                .connect(validator1)
                .validateTranslation(translationId, 60, "Poor translation");
            
            const receipt = await tx.wait();
            expect(receipt.events[0].event).to.equal("TranslationFailed");

            const translationData = await translation.getTranslation(translationId);
            expect(translationData.status).to.equal(2); // Failed
            expect(translationData.averageScore).to.equal(78); // (85 + 90 + 60) / 3
        });
    });

    describe("Score Management", function () {
        it("Should update translator score on completion", async function () {
            // Submit and complete a translation
            const tx = await translation
                .connect(translator)
                .submitTranslation(
                    "Hello, world!",
                    "¡Hola, mundo!",
                    "en",
                    "es",
                    "general"
                );
            
            const translationId = await translation.getTranslationCount();

            // Add validations
            await translation
                .connect(validator1)
                .validateTranslation(translationId, 85, "Good translation");
            await translation
                .connect(validator2)
                .validateTranslation(translationId, 90, "Very good translation");
            await translation
                .connect(validator1)
                .validateTranslation(translationId, 95, "Excellent translation");

            const translatorScore = await translation.getTranslatorScore(translator.address);
            expect(translatorScore).to.be.gt(0);
        });

        it("Should update validator score on validation", async function () {
            // Submit a translation
            const tx = await translation
                .connect(translator)
                .submitTranslation(
                    "Hello, world!",
                    "¡Hola, mundo!",
                    "en",
                    "es",
                    "general"
                );
            
            const translationId = await translation.getTranslationCount();

            // Add validation
            await translation
                .connect(validator1)
                .validateTranslation(translationId, 85, "Good translation");

            const validatorScore = await translation.getValidatorScore(validator1.address);
            expect(validatorScore).to.be.gt(0);
        });
    });
}); 