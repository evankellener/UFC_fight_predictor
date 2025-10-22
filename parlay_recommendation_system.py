"""
UFC Parlay Recommendation System

Automatically generates optimal parlay recommendations for upcoming UFC events
based on model predictions and validated strategies.
"""

import pandas as pd
import xgboost as xgb
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from datetime import timedelta
import numpy as np
from itertools import combinations

class ParlayRecommender:
    """
    Intelligent parlay recommendation system using XGBoost champion model.
    """
    
    def __init__(self, config_file='xgboost_ga_results_1760303427.json'):
        """
        Initialize the parlay recommender.
        
        Parameters:
        -----------
        config_file : str
            Path to champion model configuration
        """
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.baseline_features = self.config['features']
        self.model = None
        self.imputer = None
        self.scaler = None
        
        # Validated optimal thresholds from analysis
        self.thresholds = {
            'conservative': 0.75,   # 100% win rate on test set
            'moderate': 0.70,       # 63-74% win rate
            'aggressive': 0.65,     # 62% win rate
        }
        
        self.optimal_legs = {
            'conservative': 3,      # Best risk/reward balance
            'value': 4,            # Higher payouts
            'mega': 5,             # Massive payouts
        }
    
    def train_model(self, data_file='data/tmp/final.csv', random_seed=42):
        """
        Train the champion model on historical data.
        
        Parameters:
        -----------
        data_file : str
            Path to training data with rolling_ema
        random_seed : int
            Random seed for reproducibility
        """
        print("="*80)
        print("TRAINING PARLAY RECOMMENDATION MODEL")
        print("="*80)
        
        # Load and prepare data
        df_full = pd.read_csv(data_file, low_memory=False)
        df_full['DATE'] = pd.to_datetime(df_full['DATE'])
        
        # Add rolling_ema
        df_full = df_full.sort_values('DATE').copy()
        df_full['win_numeric'] = pd.to_numeric(df_full['win'], errors='coerce')
        df_full['rolling_ema'] = df_full['win_numeric'].ewm(span=200, min_periods=20).mean().shift(1)
        
        # Prepare data
        df = df_full.copy()
        df = df[df['DATE'] >= '2009-01-01']
        df = df[df['sex'].astype(str) == '2']
        
        # Create diff features if needed
        if 'precomp_elo_diff' not in df.columns:
            df['precomp_elo_diff'] = pd.to_numeric(df['precomp_elo'], errors='coerce') - pd.to_numeric(df['opp_precomp_elo'], errors='coerce')
        if 'precomp_strike_elo_diff' not in df.columns:
            df['precomp_strike_elo_diff'] = pd.to_numeric(df['precomp_strike_elo'], errors='coerce') - pd.to_numeric(df['opp_precomp_strike_elo'], errors='coerce')
        if 'precomp_grapple_elo_diff' not in df.columns:
            df['precomp_grapple_elo_diff'] = pd.to_numeric(df['precomp_grapple_elo'], errors='coerce') - pd.to_numeric(df['opp_precomp_grapple_elo'], errors='coerce')
        
        df = df.dropna(subset=['win'])
        df['win'] = pd.to_numeric(df['win']).astype(int)
        
        # Filter null values
        thresh = int(0.7 * len(self.baseline_features))
        null_counts = df[self.baseline_features].isnull().sum(axis=1)
        df = df[null_counts <= thresh]
        
        df = df[(pd.to_numeric(df['precomp_boutcount'], errors='coerce') >= 1) & 
                (pd.to_numeric(df['opp_precomp_boutcount'], errors='coerce') >= 1)]
        
        # Train/test split
        latest = df['DATE'].max()
        cutoff = latest - timedelta(days=365)
        train = df[df['DATE'] < cutoff]
        test = df[df['DATE'] >= cutoff]
        
        print(f"Training data: {len(train)} fights")
        print(f"Test data: {len(test)} fights")
        
        # Prepare features
        features_with_ema = self.baseline_features + ['rolling_ema']
        
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(self.imputer.fit_transform(train[features_with_ema]))
        X_test_scaled = self.scaler.transform(self.imputer.transform(test[features_with_ema]))
        
        # Train model
        self.model = xgb.XGBClassifier(random_state=random_seed, n_jobs=-1, 
                                       eval_metric='logloss', early_stopping_rounds=20, 
                                       **self.config['hyperparams'])
        
        print("\nTraining XGBoost model...")
        self.model.fit(X_train_scaled, train['win'], 
                      eval_set=[(X_test_scaled, test['win'])], verbose=False)
        
        # Validate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = (y_pred == test['win']).mean()
        
        print(f"\n✅ Model trained successfully!")
        print(f"   Validation accuracy: {accuracy*100:.2f}%")
        print(f"   Expected: ~69-71%")
        
        return self
    
    def predict_fight(self, fighter_a_stats, fighter_b_stats, rolling_ema):
        """
        Predict outcome of a single fight.
        
        Parameters:
        -----------
        fighter_a_stats : dict
            Fighter A's stats
        fighter_b_stats : dict
            Fighter B's stats
        rolling_ema : float
            Current rolling EMA value
            
        Returns:
        --------
        dict : Prediction results
        """
        # Build feature vectors
        row_a = {}
        row_b = {}
        
        features_with_ema = self.baseline_features + ['rolling_ema']
        
        for feature in features_with_ema:
            if feature == 'rolling_ema':
                row_a[feature] = rolling_ema
                row_b[feature] = rolling_ema
            elif feature.startswith('opp_'):
                base_feature = feature.replace('opp_', 'precomp_')
                row_a[feature] = fighter_b_stats.get(base_feature, np.nan)
                row_b[feature] = fighter_a_stats.get(base_feature, np.nan)
            else:
                row_a[feature] = fighter_a_stats.get(feature, np.nan)
                row_b[feature] = fighter_b_stats.get(feature, np.nan)
        
        # Predict
        X_pred = pd.DataFrame([row_a, row_b])
        X_pred_scaled = self.scaler.transform(self.imputer.transform(X_pred))
        
        probs = self.model.predict_proba(X_pred_scaled)[:, 1]
        
        # Normalize probabilities
        total = probs[0] + probs[1]
        prob_a = probs[0] / total
        prob_b = probs[1] / total
        
        return {
            'prob_a': prob_a,
            'prob_b': prob_b,
            'predicted_winner': 'A' if prob_a > 0.5 else 'B',
            'confidence': max(prob_a, prob_b)
        }
    
    def recommend_parlays(self, event_predictions, strategy='conservative', max_recommendations=10):
        """
        Generate parlay recommendations for an event.
        
        Parameters:
        -----------
        event_predictions : list of dict
            List of fight predictions with keys: 'fighter_a', 'fighter_b', 
            'predicted_winner', 'confidence', 'odds_a', 'odds_b'
        strategy : str
            'conservative', 'moderate', or 'aggressive'
        max_recommendations : int
            Maximum number of parlay recommendations to return
            
        Returns:
        --------
        list : Recommended parlays
        """
        threshold = self.thresholds[strategy]
        
        # Filter high-confidence picks
        high_conf_picks = [p for p in event_predictions if p['confidence'] >= threshold]
        
        if len(high_conf_picks) < 2:
            return {
                'message': f'Not enough high-confidence picks (need ≥2, found {len(high_conf_picks)})',
                'threshold': threshold,
                'high_confidence_fights': len(high_conf_picks),
                'parlays': []
            }
        
        # Generate parlays of different sizes
        recommendations = []
        
        # Try 3-leg, 4-leg, 5-leg parlays
        for n_legs in [3, 4, 5]:
            if len(high_conf_picks) < n_legs:
                continue
            
            # Generate all combinations
            for combo in combinations(high_conf_picks, n_legs):
                # Calculate parlay odds
                decimal_odds = []
                for pick in combo:
                    if pick['predicted_winner'] == 'A':
                        odds = pick['odds_a']
                    else:
                        odds = pick['odds_b']
                    
                    # Convert American to decimal
                    if odds > 0:
                        decimal = 1 + (odds / 100)
                    else:
                        decimal = 1 + (100 / abs(odds))
                    decimal_odds.append(decimal)
                
                parlay_decimal = np.prod(decimal_odds)
                parlay_american = (parlay_decimal - 1) * 100 if parlay_decimal >= 2 else -100 / (parlay_decimal - 1)
                
                # Calculate expected payout
                stake = 10
                payout = parlay_decimal * stake
                profit = payout - stake
                
                # Calculate average confidence
                avg_confidence = np.mean([p['confidence'] for p in combo])
                min_confidence = np.min([p['confidence'] for p in combo])
                
                recommendations.append({
                    'n_legs': n_legs,
                    'legs': [{'fighter_a': p['fighter_a'], 
                             'fighter_b': p['fighter_b'],
                             'pick': p['fighter_a'] if p['predicted_winner'] == 'A' else p['fighter_b'],
                             'confidence': p['confidence'],
                             'odds': pick['odds_a'] if p['predicted_winner'] == 'A' else p['odds_b']}
                            for p in combo],
                    'avg_confidence': avg_confidence,
                    'min_confidence': min_confidence,
                    'parlay_odds_decimal': parlay_decimal,
                    'parlay_odds_american': parlay_american,
                    'stake': stake,
                    'payout': payout,
                    'profit': profit,
                })
        
        # Sort by ROI potential (considering both confidence and payout)
        # Score = avg_confidence * profit
        for rec in recommendations:
            rec['score'] = rec['avg_confidence'] * rec['profit']
        
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        return {
            'strategy': strategy,
            'threshold': threshold,
            'total_fights': len(event_predictions),
            'high_confidence_fights': len(high_conf_picks),
            'total_parlays_generated': len(recommendations),
            'top_recommendations': recommendations[:max_recommendations]
        }
    
    def print_recommendations(self, recommendations):
        """
        Pretty print parlay recommendations.
        
        Parameters:
        -----------
        recommendations : dict
            Output from recommend_parlays()
        """
        print("="*80)
        print("PARLAY RECOMMENDATIONS")
        print("="*80)
        
        if 'message' in recommendations:
            print(f"\n⚠️  {recommendations['message']}")
            return
        
        print(f"\nStrategy: {recommendations['strategy'].upper()}")
        print(f"Confidence threshold: ≥{recommendations['threshold']*100:.0f}%")
        print(f"Total fights in event: {recommendations['total_fights']}")
        print(f"High-confidence fights: {recommendations['high_confidence_fights']}")
        print(f"Total parlays generated: {recommendations['total_parlays_generated']}")
        
        if len(recommendations['top_recommendations']) == 0:
            print("\n⚠️  No parlays meet the criteria")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {len(recommendations['top_recommendations'])} RECOMMENDED PARLAYS")
        print("="*80)
        
        for i, parlay in enumerate(recommendations['top_recommendations'], 1):
            print(f"\n{'─'*80}")
            print(f"PARLAY #{i} - {parlay['n_legs']}-LEG")
            print(f"{'─'*80}")
            print(f"Parlay Odds: {parlay['parlay_odds_american']:+.0f} (Decimal: {parlay['parlay_odds_decimal']:.2f}x)")
            print(f"Stake: ${parlay['stake']:.2f} → Payout: ${parlay['payout']:.2f} (Profit: ${parlay['profit']:+.2f})")
            print(f"Avg Confidence: {parlay['avg_confidence']*100:.1f}% | Min Confidence: {parlay['min_confidence']*100:.1f}%")
            
            print(f"\nLegs:")
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"  {j}. {leg['fighter_a']} vs {leg['fighter_b']}")
                print(f"     → PICK: {leg['pick']} ({leg['confidence']*100:.1f}% confident) at {leg['odds']:+d}")
        
        print(f"\n{'='*80}")
        print("✅ Recommendations complete!")
        print("="*80)


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("PARLAY RECOMMENDATION SYSTEM - EXAMPLE USAGE")
    print("="*80)
    
    # Initialize recommender
    recommender = ParlayRecommender()
    
    # Train model
    recommender.train_model()
    
    # Example: Create sample event predictions (normally from real data)
    print("\n" + "="*80)
    print("EXAMPLE: Upcoming UFC Event")
    print("="*80)
    
    sample_event = [
        {
            'fighter_a': 'Charles Oliveira',
            'fighter_b': 'Mateusz Gamrot',
            'predicted_winner': 'B',
            'confidence': 0.739,
            'odds_a': -115,
            'odds_b': -105,
        },
        {
            'fighter_a': 'Youssef Zalal',
            'fighter_b': 'Josh Emmett',
            'predicted_winner': 'A',
            'confidence': 0.787,
            'odds_a': -450,
            'odds_b': +350,
        },
        {
            'fighter_a': 'Veronica Hardy',
            'fighter_b': 'Brogan Walker',
            'predicted_winner': 'A',
            'confidence': 0.768,
            'odds_a': -900,
            'odds_b': +550,
        },
        {
            'fighter_a': 'Jimmy Crute',
            'fighter_b': 'Ivan Erslan',
            'predicted_winner': 'A',
            'confidence': 0.783,
            'odds_a': -250,
            'odds_b': +200,
        },
        {
            'fighter_a': 'Carlos Ulberg',
            'fighter_b': 'Dominick Reyes',
            'predicted_winner': 'A',
            'confidence': 0.747,
            'odds_a': -235,
            'odds_b': +190,
        },
    ]
    
    # Test all strategies
    for strategy in ['conservative', 'moderate', 'aggressive']:
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")
        
        recs = recommender.recommend_parlays(sample_event, strategy=strategy, max_recommendations=5)
        recommender.print_recommendations(recs)
    
    print("\n" + "="*80)
    print("PARLAY RECOMMENDATION SYSTEM READY")
    print("="*80)
    print("""
To use with your own data:

1. Create a list of fight predictions:
   predictions = [
       {'fighter_a': 'Name A', 'fighter_b': 'Name B', 
        'predicted_winner': 'A' or 'B', 'confidence': 0.XX,
        'odds_a': -XXX, 'odds_b': +XXX},
       ...
   ]

2. Get recommendations:
   recs = recommender.recommend_parlays(predictions, strategy='conservative')

3. Print results:
   recommender.print_recommendations(recs)
    """)

