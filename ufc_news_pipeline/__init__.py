"""
UFC News Pipeline - Article Scraper and Summarizer

A comprehensive toolkit for scraping and analyzing MMA news articles about UFC fights.
"""

from .article_scraper import UFCFightArticleScraper, Article, format_article_report
from .summarizer import ArticleSummarizer, create_comprehensive_summary

__all__ = [
    'UFCFightArticleScraper',
    'Article',
    'ArticleSummarizer',
    'format_article_report',
    'create_comprehensive_summary'
]

__version__ = '1.0.0'

