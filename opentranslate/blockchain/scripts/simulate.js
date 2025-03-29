const hre = require("hardhat");
const fs = require("fs");
const path = require("path");
const config = require("../config/contracts");

async function getContracts() {
    const deploymentsPath = path.join(__dirname, "..", "deployments.json");
    if (!fs.existsSync(deploymentsPath)) {
        throw new Error("No deployment info found. Please deploy contracts first.");
    }

    const deployments = JSON.parse(fs.readFileSync(deploymentsPath));
    const network = await hre.ethers.provider.getNetwork();
    
    if (deployments.network !== network.name || deployments.chainId !== network.chainId) {
        throw new Error("Network mismatch. Please ensure you're on the correct network.");
    }

    const token = await hre.ethers.getContractAt(
        "PUMPFUNToken",
        deployments.token.address
    );
    const translation = await hre.ethers.getContractAt(
        "TranslationContract",
        deployments.translation.address
    );

    return { token, translation };
}

async function setupTestAccounts() {
    console.log("\nSetting up test accounts...");
    
    const { token } = await getContracts();
    const [owner] = await hre.ethers.getSigners();
    
    // Check if signer is owner
    const tokenOwner = await token.owner();
    if (tokenOwner.toLowerCase() !== owner.address.toLowerCase()) {
        throw new Error("Only the owner can set up test accounts");
    }
    
    // Create test accounts
    const testAccounts = [];
    for (let i = 0; i < config.test.numAccounts; i++) {
        const wallet = hre.ethers.Wallet.createRandom();
        testAccounts.push(wallet);
    }
    
    // Transfer tokens to test accounts
    const transferAmount = hre.ethers.utils.parseEther(config.test.initialBalance);
    for (const account of testAccounts) {
        const tx = await token.transfer(account.address, transferAmount);
        await tx.wait();
        console.log("Transferred", config.test.initialBalance, "tokens to", account.address);
    }
    
    // Save test accounts to file
    const accountsPath = path.join(__dirname, "..", "test-accounts.json");
    const accountsData = testAccounts.map(account => ({
        address: account.address,
        privateKey: account.privateKey
    }));
    fs.writeFileSync(accountsPath, JSON.stringify(accountsData, null, 2));
    
    console.log("\nTest accounts saved to test-accounts.json");
    return testAccounts;
}

async function simulateTranslations() {
    console.log("\nSimulating translations...");
    
    const { token, translation } = await getContracts();
    
    // Load test accounts
    const accountsPath = path.join(__dirname, "..", "test-accounts.json");
    if (!fs.existsSync(accountsPath)) {
        throw new Error("Test accounts not found. Please run setup first.");
    }
    
    const testAccounts = JSON.parse(fs.readFileSync(accountsPath));
    
    // Sample translation pairs
    const translations = [
        {
            source: "Hello, how are you?",
            target: "Hola, ¿cómo estás?",
            sourceLang: "en",
            targetLang: "es",
            domain: "general"
        },
        {
            source: "The weather is nice today.",
            target: "Il fait beau aujourd'hui.",
            sourceLang: "en",
            targetLang: "fr",
            domain: "weather"
        },
        {
            source: "I love programming.",
            target: "Ich liebe Programmierung.",
            sourceLang: "en",
            targetLang: "de",
            domain: "technology"
        }
    ];
    
    // Simulate translations
    for (let i = 0; i < config.test.numTranslations; i++) {
        const account = new hre.ethers.Wallet(testAccounts[i % testAccounts.length].privateKey, hre.ethers.provider);
        const translation = translations[i % translations.length];
        
        // Stake tokens
        const stakeAmount = hre.ethers.utils.parseEther(config.test.stakeAmount);
        const stakeTx = await token.connect(account).stake(stakeAmount);
        await stakeTx.wait();
        
        // Submit translation
        const submitTx = await translation.connect(account).submitTranslation(
            translation.source,
            translation.target,
            translation.sourceLang,
            translation.targetLang,
            translation.domain
        );
        const receipt = await submitTx.wait();
        
        console.log(`Translation ${i + 1} submitted by ${account.address}`);
        console.log("Transaction hash:", receipt.transactionHash);
        
        // Wait between translations
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

async function simulateValidations() {
    console.log("\nSimulating validations...");
    
    const { token, translation } = await getContracts();
    
    // Load test accounts
    const accountsPath = path.join(__dirname, "..", "test-accounts.json");
    if (!fs.existsSync(accountsPath)) {
        throw new Error("Test accounts not found. Please run setup first.");
    }
    
    const testAccounts = JSON.parse(fs.readFileSync(accountsPath));
    
    // Get all translations
    const translationCount = await translation.getTranslationCount();
    
    // Simulate validations
    for (let i = 0; i < translationCount; i++) {
        const validationCount = await translation.getValidationCount(i);
        const minValidations = await translation.minimumValidations();
        
        if (validationCount >= minValidations) {
            continue;
        }
        
        // Get remaining validations needed
        const remainingValidations = minValidations - validationCount;
        
        // Simulate validations from different accounts
        for (let j = 0; j < remainingValidations; j++) {
            const account = new hre.ethers.Wallet(testAccounts[j % testAccounts.length].privateKey, hre.ethers.provider);
            
            // Stake tokens
            const stakeAmount = hre.ethers.utils.parseEther(config.test.stakeAmount);
            const stakeTx = await token.connect(account).stake(stakeAmount);
            await stakeTx.wait();
            
            // Generate random score between 60 and 100
            const score = Math.floor(Math.random() * 41) + 60;
            
            // Submit validation
            const validateTx = await translation.connect(account).validateTranslation(
                i,
                score,
                `Test validation ${j + 1}`
            );
            const receipt = await validateTx.wait();
            
            console.log(`Validation ${j + 1} for translation ${i} submitted by ${account.address}`);
            console.log("Score:", score);
            console.log("Transaction hash:", receipt.transactionHash);
            
            // Wait between validations
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
}

async function getSimulationStats() {
    console.log("\nSimulation Statistics:");
    
    const { token, translation } = await getContracts();
    
    // Load test accounts
    const accountsPath = path.join(__dirname, "..", "test-accounts.json");
    if (!fs.existsSync(accountsPath)) {
        throw new Error("Test accounts not found. Please run setup first.");
    }
    
    const testAccounts = JSON.parse(fs.readFileSync(accountsPath));
    
    // Get translation stats
    const translationCount = await translation.getTranslationCount();
    console.log("Total Translations:", translationCount);
    
    // Get validation stats
    let totalValidations = 0;
    for (let i = 0; i < translationCount; i++) {
        const validationCount = await translation.getValidationCount(i);
        totalValidations += validationCount;
    }
    console.log("Total Validations:", totalValidations);
    
    // Get reward stats
    const pendingRewards = await translation.getPendingRewards();
    console.log("Pending Rewards:", hre.ethers.utils.formatEther(pendingRewards), "tokens");
    
    // Get account stats
    console.log("\nAccount Statistics:");
    for (const account of testAccounts) {
        const wallet = new hre.ethers.Wallet(account.privateKey, hre.ethers.provider);
        
        const balance = await token.balanceOf(wallet.address);
        const stakedBalance = await token.stakedBalance(wallet.address);
        const translatorScore = await translation.getTranslatorScore(wallet.address);
        const validatorScore = await translation.getValidatorScore(wallet.address);
        
        console.log(`\nAccount: ${wallet.address}`);
        console.log("Balance:", hre.ethers.utils.formatEther(balance), "tokens");
        console.log("Staked:", hre.ethers.utils.formatEther(stakedBalance), "tokens");
        console.log("Translator Score:", translatorScore);
        console.log("Validator Score:", validatorScore);
    }
}

async function main() {
    try {
        const network = await hre.ethers.provider.getNetwork();
        console.log(`\nRunning simulation on network: ${network.name} (chainId: ${network.chainId})`);
        
        const command = process.argv[2];
        const args = process.argv.slice(3);
        
        switch (command) {
            case "setup":
                await setupTestAccounts();
                break;
                
            case "translate":
                await simulateTranslations();
                break;
                
            case "validate":
                await simulateValidations();
                break;
                
            case "stats":
                await getSimulationStats();
                break;
                
            case "all":
                await setupTestAccounts();
                await simulateTranslations();
                await simulateValidations();
                await getSimulationStats();
                break;
                
            default:
                console.error("Invalid command. Available commands:");
                console.error("  setup     - Set up test accounts");
                console.error("  translate - Simulate translations");
                console.error("  validate  - Simulate validations");
                console.error("  stats     - Get simulation statistics");
                console.error("  all       - Run complete simulation");
                process.exit(1);
        }
        
    } catch (error) {
        console.error("\nError:", error.message);
        process.exit(1);
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    }); 