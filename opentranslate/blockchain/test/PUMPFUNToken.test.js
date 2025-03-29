const { expect } = require("chai");
const { ethers } = require("hardhat");
const config = require("../config/contracts");

describe("PUMPFUNToken", function () {
    let token;
    let owner;
    let addr1;
    let addr2;
    let addrs;
    
    beforeEach(async function () {
        // Get signers
        [owner, addr1, addr2, ...addrs] = await ethers.getSigners();
        
        // Deploy token
        const PUMPFUNToken = await ethers.getContractFactory("PUMPFUNToken");
        token = await PUMPFUNToken.deploy();
        await token.deployed();
    });
    
    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            expect(await token.owner()).to.equal(owner.address);
        });
        
        it("Should assign the total supply to the owner", async function () {
            const ownerBalance = await token.balanceOf(owner.address);
            expect(await token.totalSupply()).to.equal(ownerBalance);
        });
        
        it("Should set the correct token name and symbol", async function () {
            expect(await token.name()).to.equal(config.token.name);
            expect(await token.symbol()).to.equal(config.token.symbol);
        });
        
        it("Should set the correct decimals", async function () {
            expect(await token.decimals()).to.equal(config.token.decimals);
        });
    });
    
    describe("Transactions", function () {
        it("Should transfer tokens between accounts", async function () {
            // Transfer 50 tokens from owner to addr1
            await token.transfer(addr1.address, 50);
            const addr1Balance = await token.balanceOf(addr1.address);
            expect(addr1Balance).to.equal(50);

            // Transfer 50 tokens from addr1 to addr2
            await token.connect(addr1).transfer(addr2.address, 50);
            const addr2Balance = await token.balanceOf(addr2.address);
            expect(addr2Balance).to.equal(50);
        });
        
        it("Should fail if sender doesn't have enough tokens", async function () {
            const initialOwnerBalance = await token.balanceOf(owner.address);
            await expect(
                token.connect(addr1).transfer(owner.address, 1)
            ).to.be.revertedWith("ERC20: transfer amount exceeds balance");
            expect(await token.balanceOf(owner.address)).to.equal(
                initialOwnerBalance
            );
        });
        
        it("Should update balances after transfers", async function () {
            const initialOwnerBalance = await token.balanceOf(owner.address);

            // Transfer 100 tokens from owner to addr1
            await token.transfer(addr1.address, 100);
            const addr1Balance = await token.balanceOf(addr1.address);
            expect(addr1Balance).to.equal(100);

            // Transfer 50 tokens from addr1 to addr2
            await token.connect(addr1).transfer(addr2.address, 50);
            const addr2Balance = await token.balanceOf(addr2.address);
            expect(addr2Balance).to.equal(50);

            const finalOwnerBalance = await token.balanceOf(owner.address);
            expect(finalOwnerBalance).to.equal(initialOwnerBalance.sub(100));
        });
    });
    
    describe("Staking", function () {
        it("Should allow users to stake tokens", async function () {
            // Transfer tokens to addr1
            await token.transfer(addr1.address, 1000);
            
            // Stake tokens
            await token.connect(addr1).stake(500);
            
            const stakedBalance = await token.stakedBalance(addr1.address);
            expect(stakedBalance).to.equal(500);
        });
        
        it("Should fail if user doesn't have enough tokens to stake", async function () {
            await expect(
                token.connect(addr1).stake(1000)
            ).to.be.revertedWith("ERC20: transfer amount exceeds balance");
        });
        
        it("Should fail if stake amount is less than minimum", async function () {
            await token.transfer(addr1.address, 1000);
            await expect(
                token.connect(addr1).stake(100)
            ).to.be.revertedWith("Stake amount must be at least minimum stake");
        });
        
        it("Should allow users to unstake tokens", async function () {
            // Transfer and stake tokens
            await token.transfer(addr1.address, 1000);
            await token.connect(addr1).stake(500);
            
            // Unstake tokens
            await token.connect(addr1).unstake(300);
            
            const stakedBalance = await token.stakedBalance(addr1.address);
            expect(stakedBalance).to.equal(200);
        });
        
        it("Should fail if user tries to unstake more than staked", async function () {
            await token.transfer(addr1.address, 1000);
            await token.connect(addr1).stake(500);
            
            await expect(
                token.connect(addr1).unstake(600)
            ).to.be.revertedWith("Insufficient staked balance");
        });
    });
    
    describe("Burning", function () {
        it("Should burn tokens on transfer", async function () {
            const initialSupply = await token.totalSupply();
            const burnAmount = ethers.utils.parseEther("100");
            
            // Transfer tokens to trigger burn
            await token.transfer(addr1.address, burnAmount);
            
            const finalSupply = await token.totalSupply();
            const expectedBurn = burnAmount.mul(config.token.burnRate).div(100);
            
            expect(finalSupply).to.equal(initialSupply.sub(expectedBurn));
        });
        
        it("Should emit Burn event", async function () {
            const burnAmount = ethers.utils.parseEther("100");
            const expectedBurn = burnAmount.mul(config.token.burnRate).div(100);
            
            await expect(token.transfer(addr1.address, burnAmount))
                .to.emit(token, "Burn")
                .withArgs(expectedBurn);
        });
    });
    
    describe("Reward Distribution", function () {
        it("Should allow reward distributor to distribute rewards", async function () {
            // Grant reward distributor role
            const rewardDistributorRole = await token.REWARD_DISTRIBUTOR_ROLE();
            await token.grantRole(rewardDistributorRole, addr1.address);
            
            // Transfer tokens to contract for rewards
            await token.transfer(token.address, 1000);
            
            // Distribute rewards
            await token.connect(addr1).distributeRewards(addr2.address, 100);
            
            const balance = await token.balanceOf(addr2.address);
            expect(balance).to.equal(100);
        });
        
        it("Should fail if non-reward distributor tries to distribute rewards", async function () {
            await token.transfer(token.address, 1000);
            
            await expect(
                token.connect(addr1).distributeRewards(addr2.address, 100)
            ).to.be.revertedWith("Caller is not a reward distributor");
        });
    });
    
    describe("Pausing", function () {
        it("Should allow owner to pause and unpause", async function () {
            await token.pause();
            expect(await token.paused()).to.be.true;
            
            await token.unpause();
            expect(await token.paused()).to.be.false;
        });
        
        it("Should fail if non-owner tries to pause", async function () {
            await expect(
                token.connect(addr1).pause()
            ).to.be.revertedWith("Ownable: caller is not the owner");
        });
        
        it("Should prevent transfers when paused", async function () {
            await token.pause();
            await expect(
                token.transfer(addr1.address, 100)
            ).to.be.revertedWith("Pausable: paused");
        });
    });
}); 