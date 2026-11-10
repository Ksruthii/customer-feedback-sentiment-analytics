SELECT complaint_theme, sentiment, COUNT(*) AS feedback_count
FROM customer_feedback_classified
GROUP BY complaint_theme, sentiment
ORDER BY feedback_count DESC;

SELECT complaint_theme, COUNT(*) AS negative_feedback_count
FROM customer_feedback_classified
WHERE sentiment = 'negative'
GROUP BY complaint_theme
ORDER BY negative_feedback_count DESC;
