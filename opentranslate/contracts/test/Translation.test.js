const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Translation", function () {
    let Translation;
    let PUMPFUNToken;
    let translation;
    let pumpfunToken;
    let owner;
    let translator;
    let validator;
    let addrs;

    beforeEach(async function () {
        [owner, translator, validator, ...addrs] = await ethers.getSigners();

        // Deploy PUMPFUN token
        PUMPFUNToken = await ethers.getContractFactory("PUMPFUNToken");
        pumpfunToken = await PUMPFUNToken.deploy();
        await pumpfunToken.deployed();

        // Deploy Translation contract
        Translation = await ethers.getContractFactory("Translation");
        translation = await Translation.deploy(pumpfunToken.address);
        await translation.deployed();

        // Add Translation contract as minter
        await pumpfunToken.addMinter(translation.address);

        // Transfer tokens to translator for staking
        const amount = ethers.utils.parseEther("2000");
        await pumpfunToken.transfer(translator.address, amount);
        await pumpfunToken.connect(translator).approve(translation.address, amount);
    });

    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            expect(await translation.owner()).to.equal(owner.address);
        });

        it("Should set the right token", async function () {
            expect(await translation.pumpfunToken()).to.equal(pumpfunToken.address);
        });
    });

    describe("Staking", function () {
        it("Should allow staking tokens", async function () {
            const stakeAmount = ethers.utils.parseEther("1000");
            await translation.connect(translator).depositStake(stakeAmount);

            const translatorInfo = await translation.getTranslator(translator.address);
            expect(translatorInfo.stake).to.equal(stakeAmount);
            expect(translatorInfo.isActive).to.equal(true);
        });

        it("Should fail if stake amount is too low", async function () {
            const lowStake = ethers.utils.parseEther("100");
            await expect(
                translation.connect(translator).depositStake(lowStake)
            ).to.be.revertedWith("Stake amount too low");
        });

        it("Should allow withdrawing stake", async function () {
            const stakeAmount = ethers.utils.parseEther("1500");
            await translation.connect(translator).depositStake(stakeAmount);

            const withdrawAmount = ethers.utils.parseEther("500");
            await translation.connect(translator).withdrawStake(withdrawAmount);

            const translatorInfo = await translation.getTranslator(translator.address);
            expect(translatorInfo.stake).to.equal(stakeAmount.sub(withdrawAmount));
            expect(translatorInfo.isActive).to.equal(true);
        });
    });

    describe("Translation Management", function () {
        beforeEach(async function () {
            const stakeAmount = ethers.utils.parseEther("1000");
            await translation.connect(translator).depositStake(stakeAmount);
        });

        it("Should record a translation", async function () {
            await translation.connect(translator).recordTranslation(
                "test-id",
                "en",
                "es",
                "general"
            );

            const record = await translation.getTranslation("test-id");
            expect(record.translator).to.equal(translator.address);
            expect(record.sourceLanguage).to.equal("en");
            expect(record.targetLanguage).to.equal("es");
            expect(record.domain).to.equal("general");
        });

        it("Should validate a translation", async function () {
            await translation.connect(translator).recordTranslation(
                "test-id",
                "en",
                "es",
                "general"
            );

            await translation.connect(validator).validateTranslation(
                "test-id",
                80
            );

            const record = await translation.getTranslation("test-id");
            expect(record.validated).to.equal(true);
            expect(record.score).to.equal(80);
        });

        it("Should not allow self-validation", async function () {
            await translation.connect(translator).recordTranslation(
                "test-id",
                "en",
                "es",
                "general"
            );

            await expect(
                translation.connect(translator).validateTranslation("test-id", 80)
            ).to.be.revertedWith("Cannot validate own translation");
        });
    });

    describe("Rewards", function () {
        beforeEach(async function () {
            const stakeAmount = ethers.utils.parseEther("1000");
            await translation.connect(translator).depositStake(stakeAmount);

            await translation.connect(translator).recordTranslation(
                "test-id",
                "en",
                "es",
                "general"
            );

            await translation.connect(validator).validateTranslation(
                "test-id",
                80
            );
        });

        it("Should accumulate rewards", async function () {
            const translatorInfo = await translation.getTranslator(translator.address);
            expect(translatorInfo.totalRewards).to.be.gt(0);
        });

        it("Should allow claiming rewards", async function () {
            const beforeBalance = await pumpfunToken.balanceOf(translator.address);
            await translation.connect(translator).claimRewards();
            const afterBalance = await pumpfunToken.balanceOf(translator.address);

            expect(afterBalance).to.be.gt(beforeBalance);

            const translatorInfo = await translation.getTranslator(translator.address);
            expect(translatorInfo.totalRewards).to.equal(0);
        });
    });
}); 