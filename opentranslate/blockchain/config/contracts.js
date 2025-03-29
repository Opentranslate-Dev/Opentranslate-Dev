module.exports = {
    token: {
        name: "PUMPFUN Token",
        symbol: "PUMPFUN",
        initialSupply: "1000000000", // 1 billion tokens
        burnRate: 2, // 2% burn rate
        minimumStake: "1000", // Minimum stake amount
        decimals: 18,
        roles: {
            DEFAULT_ADMIN: "0x0000000000000000000000000000000000000000000000000000000000000000",
            REWARD_DISTRIBUTOR: "0x0000000000000000000000000000000000000000000000000000000000000001",
        },
    },
    translation: {
        minimumStake: "1000", // Minimum stake for translators
        minimumValidations: 3, // Minimum number of validations required
        minimumScore: 70, // Minimum score for completion (70%)
        baseReward: "100", // Base reward for translation
        goodScoreMultiplier: 120, // Good score multiplier (120%)
        poorScoreMultiplier: 80, // Poor score multiplier (80%)
        validatorRewardMultiplier: 2, // Validator reward multiplier
        treasury: {
            initialAmount: "100000000", // 100M tokens for treasury
        },
        rewards: {
            contractAmount: "50000000", // 50M tokens for contract rewards
        },
        testAccounts: {
            count: 5, // Number of test accounts to setup
            stakeAmount: "1000", // Amount to stake for each test account
        },
    },
    networks: {
        hardhat: {
            chainId: 31337,
        },
        localhost: {
            url: "http://127.0.0.1:8545",
            chainId: 31337,
        },
        goerli: {
            url: process.env.GOERLI_URL || "",
            chainId: 5,
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
        },
        sepolia: {
            url: process.env.SEPOLIA_URL || "",
            chainId: 11155111,
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
        },
        mainnet: {
            url: process.env.MAINNET_URL || "",
            chainId: 1,
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
        },
    },
    etherscan: {
        apiKey: process.env.ETHERSCAN_API_KEY,
    },
}; 