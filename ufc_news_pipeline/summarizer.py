"""
Article Summarization Module
Provides multiple summarization strategies for MMA news articles
"""

import re
from typing import List, Dict, Optional
from collections import Counter
import math


class ArticleSummarizer:
    """
    Provides various summarization methods for articles
    
    Methods:
    - Extractive (sentence selection based on importance)
    - Keyword-based (focusing on fight-relevant content)
    - Fighter-focused (emphasizing fighter mentions)
    """
    
    def __init__(self):
        # Important keywords for fight analysis
        self.fight_keywords = {
            'high': ['victory', 'defeated', 'knockout', 'submission', 'decision', 
                    'champion', 'title', 'main event', 'injury', 'camp', 'training'],
            'medium': ['fight', 'bout', 'match', 'round', 'strike', 'grappling',
                      'preparation', 'weigh-in', 'weight', 'odds', 'favorite'],
            'low': ['ufc', 'mma', 'fighter', 'coach', 'corner', 'cage']
        }
        
        # Negative indicators (sentences to potentially skip)
        self.skip_indicators = [
            'advertisement', 'subscribe', 'follow us', 'click here',
            'next page', 'related articles', 'trending now'
        ]
    
    def extractive_summary(self, text: str, num_sentences: int = 5,
                          fighter_names: Optional[List[str]] = None) -> str:
        """
        Create an extractive summary by selecting the most important sentences
        
        Args:
            text: Full article text
            num_sentences: Number of sentences to include in summary
            fighter_names: Optional list of fighter names to prioritize
            
        Returns:
            Summary string
        """
        if not text or len(text) < 100:
            return text
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Score each sentence
        scores = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences), fighter_names)
            scores.append((score, i, sentence))
        
        # Sort by score and select top sentences
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Take top N sentences, but preserve original order
        selected_indices = sorted([s[1] for s in scores[:num_sentences]])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        return ' '.join(summary_sentences)
    
    def keyword_based_summary(self, text: str, num_sentences: int = 5,
                             fighter_names: Optional[List[str]] = None) -> str:
        """
        Create a summary focusing on fight-relevant keywords
        
        Args:
            text: Full article text
            num_sentences: Number of sentences to include
            fighter_names: Optional list of fighter names
            
        Returns:
            Summary string
        """
        if not text or len(text) < 100:
            return text
        
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Score based on keyword relevance
        scored = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # High-value keywords
            for keyword in self.fight_keywords['high']:
                if keyword in sentence_lower:
                    score += 3
            
            # Medium-value keywords
            for keyword in self.fight_keywords['medium']:
                if keyword in sentence_lower:
                    score += 2
            
            # Low-value keywords
            for keyword in self.fight_keywords['low']:
                if keyword in sentence_lower:
                    score += 1
            
            # Bonus for fighter names
            if fighter_names:
                for fighter in fighter_names:
                    if fighter.lower() in sentence_lower:
                        score += 4
            
            # Penalty for skip indicators
            for indicator in self.skip_indicators:
                if indicator in sentence_lower:
                    score -= 10
            
            # Slight bonus for position (earlier sentences often more important)
            position_bonus = (len(sentences) - i) / len(sentences)
            score += position_bonus
            
            scored.append((score, i, sentence))
        
        # Sort and select
        scored.sort(reverse=True, key=lambda x: x[0])
        selected_indices = sorted([s[1] for s in scored[:num_sentences]])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        return ' '.join(summary_sentences)
    
    def fighter_focused_summary(self, text: str, fighter_names: List[str],
                               num_sentences: int = 5) -> Dict[str, str]:
        """
        Create separate summaries for each fighter
        
        Args:
            text: Full article text
            fighter_names: List of fighter names
            num_sentences: Number of sentences per fighter
            
        Returns:
            Dictionary mapping fighter names to their summaries
        """
        sentences = self._split_sentences(text)
        
        summaries = {}
        for fighter in fighter_names:
            # Find sentences mentioning this fighter
            fighter_sentences = []
            for sentence in sentences:
                if fighter.lower() in sentence.lower():
                    fighter_sentences.append(sentence)
            
            # Create summary from fighter-specific sentences
            if fighter_sentences:
                if len(fighter_sentences) <= num_sentences:
                    summaries[fighter] = ' '.join(fighter_sentences)
                else:
                    # Score and select most important
                    scored = []
                    for i, sent in enumerate(fighter_sentences):
                        score = self._score_sentence(sent, i, len(fighter_sentences), [fighter])
                        scored.append((score, i, sent))
                    
                    scored.sort(reverse=True, key=lambda x: x[0])
                    selected = [s[2] for s in scored[:num_sentences]]
                    summaries[fighter] = ' '.join(selected)
            else:
                summaries[fighter] = f"No specific information about {fighter} found in this article."
        
        return summaries
    
    def bullet_point_summary(self, text: str, num_points: int = 5,
                            fighter_names: Optional[List[str]] = None) -> List[str]:
        """
        Create a bullet-point summary with key facts
        
        Args:
            text: Full article text
            num_points: Number of bullet points
            fighter_names: Optional list of fighter names
            
        Returns:
            List of bullet point strings
        """
        sentences = self._split_sentences(text)
        
        # Score and select diverse sentences
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences), fighter_names)
            
            # Prefer shorter, punchier sentences for bullets
            word_count = len(sentence.split())
            if 10 <= word_count <= 25:  # Sweet spot for bullet points
                score += 2
            elif word_count > 40:  # Too long
                score -= 1
            
            scored.append((score, sentence))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Select top sentences and clean them up
        bullets = []
        for score, sentence in scored[:num_points]:
            # Remove leading articles and clean up
            bullet = sentence.strip()
            if bullet.startswith(('The ', 'A ', 'An ')):
                bullet = bullet.split(' ', 1)[1]
            bullets.append(bullet)
        
        return bullets
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences
    
    def _score_sentence(self, sentence: str, position: int, total: int,
                       fighter_names: Optional[List[str]] = None) -> float:
        """
        Score a sentence for importance
        
        Args:
            sentence: The sentence to score
            position: Position in document (0-indexed)
            total: Total number of sentences
            fighter_names: Optional fighter names to boost score
            
        Returns:
            Importance score (higher = more important)
        """
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Position-based score (first and last sentences often important)
        if position < 3:  # First few sentences
            score += 3 - position
        elif position >= total - 2:  # Last few sentences
            score += 1
        
        # Length-based score (prefer medium-length sentences)
        word_count = len(sentence.split())
        if 15 <= word_count <= 30:
            score += 2
        elif word_count < 10 or word_count > 50:
            score -= 1
        
        # Keyword presence
        for keyword in self.fight_keywords['high']:
            if keyword in sentence_lower:
                score += 2
        
        for keyword in self.fight_keywords['medium']:
            if keyword in sentence_lower:
                score += 1
        
        # Fighter name mentions
        if fighter_names:
            for fighter in fighter_names:
                if fighter.lower() in sentence_lower:
                    score += 3
        
        # Penalty for skip indicators
        for indicator in self.skip_indicators:
            if indicator in sentence_lower:
                score -= 5
        
        # Numbers and specific details are often important
        if re.search(r'\d+', sentence):
            score += 0.5
        
        # Quotes are often important
        if '"' in sentence or "'" in sentence:
            score += 1
        
        return score
    
    def get_key_points(self, text: str, fighter_names: List[str]) -> Dict[str, any]:
        """
        Extract key points and information from the article
        
        Args:
            text: Full article text
            fighter_names: List of fighter names
            
        Returns:
            Dictionary with structured key points
        """
        text_lower = text.lower()
        
        key_points = {
            'injury_concerns': [],
            'training_updates': [],
            'predictions': [],
            'fight_details': [],
            'quotes': []
        }
        
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            
            # Injury-related
            if any(word in sent_lower for word in ['injury', 'injured', 'hurt', 'damage']):
                key_points['injury_concerns'].append(sentence)
            
            # Training/camp-related
            if any(word in sent_lower for word in ['training', 'camp', 'preparation', 'sparring']):
                key_points['training_updates'].append(sentence)
            
            # Predictions
            if any(word in sent_lower for word in ['predict', 'forecast', 'expect', 'likely', 'favorite']):
                key_points['predictions'].append(sentence)
            
            # Fight details
            if any(word in sent_lower for word in ['main event', 'co-main', 'card', 'venue', 'date']):
                key_points['fight_details'].append(sentence)
            
            # Quotes (contains quotation marks)
            if '"' in sentence:
                key_points['quotes'].append(sentence)
        
        # Limit each category
        for key in key_points:
            key_points[key] = key_points[key][:3]  # Keep top 3 of each
        
        return key_points


def create_comprehensive_summary(text: str, fighter_names: List[str],
                                 include_bullets: bool = True) -> str:
    """
    Create a comprehensive summary with multiple formats
    
    Args:
        text: Full article text
        fighter_names: List of fighter names
        include_bullets: Whether to include bullet points
        
    Returns:
        Formatted comprehensive summary
    """
    summarizer = ArticleSummarizer()
    
    output = []
    output.append("=" * 80)
    output.append("COMPREHENSIVE ARTICLE SUMMARY")
    output.append("=" * 80)
    
    # Main summary
    output.append("\nMAIN SUMMARY:")
    main_summary = summarizer.extractive_summary(text, num_sentences=5, 
                                                 fighter_names=fighter_names)
    output.append(main_summary)
    
    # Bullet points
    if include_bullets:
        output.append("\n\nKEY POINTS:")
        bullets = summarizer.bullet_point_summary(text, num_points=5,
                                                  fighter_names=fighter_names)
        for bullet in bullets:
            output.append(f"  • {bullet}")
    
    # Fighter-specific summaries
    output.append("\n\nFIGHTER-SPECIFIC INFORMATION:")
    fighter_summaries = summarizer.fighter_focused_summary(text, fighter_names,
                                                           num_sentences=3)
    for fighter, summary in fighter_summaries.items():
        output.append(f"\n{fighter.upper()}:")
        output.append(f"  {summary}")
    
    # Key points
    output.append("\n\nCATEGORIZED INFORMATION:")
    key_points = summarizer.get_key_points(text, fighter_names)
    
    for category, points in key_points.items():
        if points:
            category_name = category.replace('_', ' ').title()
            output.append(f"\n{category_name}:")
            for point in points:
                output.append(f"  • {point[:150]}...")  # Truncate long sentences
    
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Jon Jones has confirmed his training camp is going well ahead of his 
    highly anticipated heavyweight title defense against Stipe Miocic at UFC 295. 
    The former light heavyweight champion has been training in Albuquerque with 
    his longtime coach Greg Jackson. Jones suffered a minor injury earlier in camp 
    but says he is now fully recovered and ready for the November showdown. 
    
    Miocic, the former heavyweight champion, has been preparing at his gym in Cleveland. 
    The 41-year-old firefighter is coming off a long layoff but insists he is in the 
    best shape of his life. Many analysts predict Jones will win by submission, 
    but Miocic's knockout power cannot be discounted.
    
    The main event takes place at Madison Square Garden on November 16th. This will be 
    Jones' first title defense after winning the vacant heavyweight belt earlier this year. 
    Both fighters made weight successfully, with Jones weighing in at 239 pounds and 
    Miocic at 241 pounds.
    """
    
    summarizer = ArticleSummarizer()
    fighters = ["Jon Jones", "Stipe Miocic"]
    
    print("EXTRACTIVE SUMMARY:")
    print(summarizer.extractive_summary(sample_text, num_sentences=3, fighter_names=fighters))
    
    print("\n\nKEYWORD-BASED SUMMARY:")
    print(summarizer.keyword_based_summary(sample_text, num_sentences=3, fighter_names=fighters))
    
    print("\n\nBULLET POINTS:")
    bullets = summarizer.bullet_point_summary(sample_text, num_points=5, fighter_names=fighters)
    for bullet in bullets:
        print(f"  • {bullet}")
    
    print("\n\nFIGHTER-FOCUSED SUMMARIES:")
    fighter_sums = summarizer.fighter_focused_summary(sample_text, fighters, num_sentences=2)
    for fighter, summary in fighter_sums.items():
        print(f"\n{fighter}:")
        print(f"  {summary}")
    
    print("\n\n" + "=" * 80)
    print(create_comprehensive_summary(sample_text, fighters))

