// UFC Fight Predictor JavaScript

class UFCFightPredictor {
    constructor() {
        this.fighters = [];
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.loadFighters();
        this.setDefaultDate();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // Fighter input with autocomplete
        this.setupFighterInput('fighter1');
        this.setupFighterInput('fighter2');

        // Search buttons
        document.getElementById('fighter1Search').addEventListener('click', () => {
            this.searchFighter('fighter1');
        });

        document.getElementById('fighter2Search').addEventListener('click', () => {
            this.searchFighter('fighter2');
        });
    }

    setupFighterInput(inputId) {
        const input = document.getElementById(inputId);
        const suggestions = document.getElementById(inputId + 'Suggestions');

        input.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            if (query.length >= 2) {
                this.showSuggestions(inputId, query);
            } else {
                this.hideSuggestions(inputId);
            }
        });

        input.addEventListener('blur', () => {
            // Delay hiding suggestions to allow clicking on them
            setTimeout(() => this.hideSuggestions(inputId), 200);
        });

        input.addEventListener('focus', () => {
            if (input.value.length >= 2) {
                this.showSuggestions(inputId, input.value.toLowerCase());
            }
        });
    }

    async loadFighters() {
        try {
            const response = await fetch('/fighters');
            const data = await response.json();
            
            if (data.fighters) {
                this.fighters = data.fighters;
                console.log(`Loaded ${this.fighters.length} fighters`);
            } else {
                this.showError('Failed to load fighters list');
            }
        } catch (error) {
            console.error('Error loading fighters:', error);
            this.showError('Failed to load fighters list');
        }
    }

    setDefaultDate() {
        const today = new Date();
        const tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);
        
        document.getElementById('fightDate').value = tomorrow.toISOString().split('T')[0];
    }

    showSuggestions(inputId, query) {
        const suggestions = document.getElementById(inputId + 'Suggestions');
        const matchingFighters = this.fighters.filter(fighter => 
            fighter.toLowerCase().includes(query)
        ).slice(0, 10);

        if (matchingFighters.length > 0) {
            suggestions.innerHTML = matchingFighters.map(fighter => 
                `<div class="suggestion-item" data-fighter="${fighter}">${fighter}</div>`
            ).join('');

            // Add click listeners to suggestions
            suggestions.querySelectorAll('.suggestion-item').forEach(item => {
                item.addEventListener('click', () => {
                    document.getElementById(inputId).value = item.dataset.fighter;
                    this.hideSuggestions(inputId);
                });
            });

            suggestions.style.display = 'block';
        } else {
            this.hideSuggestions(inputId);
        }
    }

    hideSuggestions(inputId) {
        document.getElementById(inputId + 'Suggestions').style.display = 'none';
    }

    async searchFighter(inputId) {
        const input = document.getElementById(inputId);
        const query = input.value.trim();
        
        if (query.length < 2) {
            this.showError('Please enter at least 2 characters to search');
            return;
        }

        const matchingFighters = this.fighters.filter(fighter => 
            fighter.toLowerCase().includes(query.toLowerCase())
        );

        if (matchingFighters.length === 0) {
            this.showError(`No fighters found matching "${query}"`);
        } else if (matchingFighters.length === 1) {
            input.value = matchingFighters[0];
        } else {
            this.showSuggestions(inputId, query.toLowerCase());
        }
    }

    async handlePrediction() {
        const fighter1 = document.getElementById('fighter1').value.trim();
        const fighter2 = document.getElementById('fighter2').value.trim();
        const fightDate = document.getElementById('fightDate').value;

        if (!fighter1 || !fighter2) {
            this.showError('Please enter both fighter names');
            return;
        }

        if (fighter1.toLowerCase() === fighter2.toLowerCase()) {
            this.showError('Please select two different fighters');
            return;
        }

        this.showLoading();
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fighter1: fighter1,
                    fighter2: fighter2,
                    fight_date: fightDate
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.displayResults(data);
            } else {
                this.showError(data.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Failed to get prediction. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(data) {
        const resultsCard = document.getElementById('resultsCard');
        const resultsBody = document.getElementById('resultsBody');
        const statsComparison = document.getElementById('statsComparison');

        // Create results HTML
        const resultsHTML = `
            <div class="prediction-result fade-in-up">
                <div class="winner-name">
                    <i class="fas fa-trophy"></i> ${data.predicted_winner}
                </div>
                <div class="confidence-display">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                    </div>
                    <p class="text-center mt-2">
                        <strong>Confidence: ${(data.confidence * 100).toFixed(1)}%</strong>
                    </p>
                </div>
                <div class="probability-display">
                    <div class="probability-item">
                        <div class="probability-value">${(data.fighter1_win_probability * 100).toFixed(1)}%</div>
                        <div class="probability-label">${data.fighter1}</div>
                    </div>
                    <div class="probability-item">
                        <div class="probability-value">${(data.fighter2_win_probability * 100).toFixed(1)}%</div>
                        <div class="probability-label">${data.fighter2}</div>
                    </div>
                </div>
                <div class="fight-details mt-3">
                    <small class="text-muted">
                        <i class="fas fa-calendar"></i> Fight Date: ${data.fight_date}
                    </small>
                </div>
            </div>
        `;

        resultsBody.innerHTML = resultsHTML;
        resultsCard.style.display = 'block';

        // Display fighter stats comparison
        if (data.fighter1_stats && data.fighter2_stats) {
            this.displayFighterStats(data.fighter1_stats, data.fighter2_stats);
            statsComparison.style.display = 'block';
        }

        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    displayFighterStats(fighter1Stats, fighter2Stats) {
        const fighterStats = document.getElementById('fighterStats');
        
        const statsHTML = `
            <div class="col-md-5">
                <div class="fighter-card">
                    <h4>${fighter1Stats.name}</h4>
                    <div class="fighter-stats">
                        <div class="stat-item">
                            <span class="stat-label">Age:</span>
                            <span class="stat-value">${fighter1Stats.age || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Height:</span>
                            <span class="stat-value">${fighter1Stats.height || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Weight:</span>
                            <span class="stat-value">${fighter1Stats.weight || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Reach:</span>
                            <span class="stat-value">${fighter1Stats.reach || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Stance:</span>
                            <span class="stat-value">${fighter1Stats.stance || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Elo Rating:</span>
                            <span class="stat-value">${fighter1Stats.precomp_elo || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Record:</span>
                            <span class="stat-value">${fighter1Stats.wins}-${fighter1Stats.losses}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Total Fights:</span>
                            <span class="stat-value">${fighter1Stats.total_fights}</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-2 text-center">
                <div class="vs-divider">VS</div>
            </div>
            <div class="col-md-5">
                <div class="fighter-card">
                    <h4>${fighter2Stats.name}</h4>
                    <div class="fighter-stats">
                        <div class="stat-item">
                            <span class="stat-label">Age:</span>
                            <span class="stat-value">${fighter2Stats.age || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Height:</span>
                            <span class="stat-value">${fighter2Stats.height || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Weight:</span>
                            <span class="stat-value">${fighter2Stats.weight || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Reach:</span>
                            <span class="stat-value">${fighter2Stats.reach || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Stance:</span>
                            <span class="stat-value">${fighter2Stats.stance || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Elo Rating:</span>
                            <span class="stat-value">${fighter2Stats.precomp_elo || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Record:</span>
                            <span class="stat-value">${fighter2Stats.wins}-${fighter2Stats.losses}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Total Fights:</span>
                            <span class="stat-value">${fighter2Stats.total_fights}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        fighterStats.innerHTML = statsHTML;
    }

    showLoading() {
        document.getElementById('loadingCard').style.display = 'block';
        document.getElementById('resultsCard').style.display = 'none';
        document.getElementById('statsComparison').style.display = 'none';
        document.getElementById('predictBtn').disabled = true;
    }

    hideLoading() {
        document.getElementById('loadingCard').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
    }

    showError(message) {
        const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
        document.getElementById('errorMessage').textContent = message;
        errorModal.show();
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new UFCFightPredictor();
});
