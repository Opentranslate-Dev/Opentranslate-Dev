const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PUMPFUNToken", function () {
    let PUMPFUNToken;
    let token;
    let owner;
    let minter;
    let recipient;
    let addrs;

    beforeEach(async function () {
        [owner, minter, recipient, ...addrs] = await ethers.getSigners();

        PUMPFUNToken = await ethers.getContractFactory("PUMPFUNToken");
        token = await PUMPFUNToken.deploy();
        await token.deployed();
    });

    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            expect(await token.owner()).to.equal(owner.address);
        });

        it("Should assign the total supply of tokens to the owner", async function () {
            const ownerBalance = await token.balanceOf(owner.address);
            expect(await token.totalSupply()).to.equal(ownerBalance);
        });

        it("Should set the correct initial supply", async function () {
            const expectedSupply = ethers.utils.parseEther("1000000000"); // 1 billion tokens
            expect(await token.totalSupply()).to.equal(expectedSupply);
        });
    });

    describe("Minting", function () {
        beforeEach(async function () {
            await token.addMinter(minter.address);
        });

        it("Should allow minter to mint tokens", async function () {
            const mintAmount = ethers.utils.parseEther("1000");
            await token.connect(minter).mint(recipient.address, mintAmount);
            expect(await token.balanceOf(recipient.address)).to.equal(mintAmount);
        });

        it("Should fail if non-minter tries to mint", async function () {
            const mintAmount = ethers.utils.parseEther("1000");
            await expect(
                token.connect(recipient).mint(recipient.address, mintAmount)
            ).to.be.revertedWith("Caller is not a minter");
        });
    });

    describe("Burning", function () {
        it("Should burn tokens on transfer", async function () {
            const transferAmount = ethers.utils.parseEther("1000");
            const burnRate = await token.BURN_RATE();
            const expectedBurn = transferAmount.mul(burnRate).div(100);

            await token.transfer(recipient.address, transferAmount);

            expect(await token.balanceOf(recipient.address)).to.equal(
                transferAmount.sub(expectedBurn)
            );
        });

        it("Should allow token holder to burn their tokens", async function () {
            const initialAmount = ethers.utils.parseEther("1000");
            const burnAmount = ethers.utils.parseEther("100");

            await token.transfer(recipient.address, initialAmount);
            await token.connect(recipient).burn(burnAmount);

            expect(await token.balanceOf(recipient.address)).to.equal(
                initialAmount.sub(burnAmount)
            );
        });
    });

    describe("Pausing", function () {
        it("Should pause and unpause", async function () {
            await token.pause();
            expect(await token.paused()).to.equal(true);

            await token.unpause();
            expect(await token.paused()).to.equal(false);
        });

        it("Should not allow transfers when paused", async function () {
            await token.pause();
            await expect(
                token.transfer(recipient.address, 100)
            ).to.be.revertedWith("Pausable: paused");
        });

        it("Should not allow non-owner to pause", async function () {
            await expect(
                token.connect(recipient).pause()
            ).to.be.revertedWith("Ownable: caller is not the owner");
        });
    });

    describe("Minter Management", function () {
        it("Should add and remove minters", async function () {
            await token.addMinter(minter.address);
            expect(await token.minters(minter.address)).to.equal(true);

            await token.removeMinter(minter.address);
            expect(await token.minters(minter.address)).to.equal(false);
        });

        it("Should not allow adding zero address as minter", async function () {
            await expect(
                token.addMinter(ethers.constants.AddressZero)
            ).to.be.revertedWith("Invalid minter address");
        });

        it("Should not allow non-owner to add minter", async function () {
            await expect(
                token.connect(recipient).addMinter(minter.address)
            ).to.be.revertedWith("Ownable: caller is not the owner");
        });
    });

    describe("Token Information", function () {
        it("Should return the correct name and symbol", async function () {
            expect(await token.name()).to.equal("PUMPFUN Token");
            expect(await token.symbol()).to.equal("PUMPFUN");
        });

        it("Should return the correct decimals", async function () {
            expect(await token.decimals()).to.equal(18);
        });
    });
}); 