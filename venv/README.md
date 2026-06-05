<p align="center">
  <strong style="font-size: 28px; color: #00F5C4;">VALERIX</strong><br/>
  <span style="font-size: 18px;">Online Payment Fraud Detection System</span>
</p>

<p align="center">
  <em>AI-powered fraud detection that identifies suspicious transactions before financial damage occurs.</em>
</p>

<p align="center">
  Python · Machine Learning · Fraud Analytics · Risk Intelligence
</p>

---

## Overview

**Valerix Online Payment Fraud Detection System** is an end-to-end machine learning solution designed to identify fraudulent online payment transactions with high accuracy. The system simulates real-world financial transactions, engineers fraud-focused features, trains multiple machine learning models, and generates professional visual analytics dashboards.

The project demonstrates how Artificial Intelligence can strengthen financial security by detecting hidden fraud patterns in transaction behavior, transaction amounts, account balances, and customer activity.

---

## Why This Project?

Online payment platforms process millions of transactions every day. Traditional rule-based fraud systems often struggle to detect sophisticated fraud attempts.

Valerix combines:

- Intelligent feature engineering
- Advanced machine learning algorithms
- Fraud probability scoring
- Risk visualization dashboards
- Performance comparison across models

to provide a comprehensive fraud detection framework.

---

## Key Features

### Fraud Detection Engine

- Detects suspicious payment transactions
- Analyzes transaction amount patterns
- Identifies high-risk transaction types
- Monitors unusual account balance changes
- Detects abnormal night-time transactions

### Machine Learning Models

The system trains and compares:

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression

The best-performing model is automatically selected using ROC-AUC score.

### Advanced Feature Engineering

The system creates intelligent fraud indicators such as:

- Balance Difference (Origin)
- Balance Difference (Destination)
- Amount-to-Balance Ratio
- Large Transaction Flags
- Night Transaction Detection
- Zero Balance Indicators

### Executive Analytics Dashboard

Generates six professional visual reports automatically.

---

## System Workflow

```text
Transaction Data
       │
       ▼
┌──────────────────────┐
│ Data Generation      │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Feature Engineering  │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Data Balancing       │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Model Training       │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Performance Analysis │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Fraud Prediction     │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Visual Dashboards    │
└──────────────────────┘