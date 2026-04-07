# Project2
# 🧾 AI Contract Clause Builder

A Streamlit application that **automatically drafts, reviews, and refines legal contract clauses** based on user-defined objectives, jurisdiction, and uploaded legal references.

👉 **Live Demo:** [Streamlit Cloud App](https://miaproject2.streamlit.app/)

---
基于法律行业用户制作简单合同文书的时间成本痛点，和非法律行业用户对便宜简单法律文书的需求痛点，制作的AI合同拟制助手。它的功能是依据用户输入的需求和相关法域，写出可下载的文本格式的法律文书。这是一个0-1的AI实践项目，我主要参与了用户的需求调研、产品框架设计、核心功能搭建与网页部署。早期是个人拉动深圳的初创企业作为技术支持，参与“互联网+”国家级比赛并获得省级铜奖。后来，自己从0通过vibe coding，接入API并进行模型预训练，实现87%的结果准召和50%的效率提升，并搭建前端网页。

## 🚀 Application Overview

This app allows users to:
- Input a **contract clause drafting objective**
- Select a **jurisdiction and drafting style**
- Upload **reference documents** (e.g., legal notes, precedents)
- Automatically generate, review, and improve legal clauses
- Download the final clause as a **Word document**

---

## 🧪 Demonstration Test Cases

Below are four test cases to demonstrate the application’s functionality.

---

### **Test Case 1: Liability Limitation (Basic Test – No Documents)**

**Objective:**  
Limit liability for indirect, special, or consequential damages to a maximum of 20% of the total contract amount, and specify exclusions from liability.

**Jurisdiction:**  
United States

**Uploaded Documents:**  
_None_

**Drafting Style:**  
Balanced (Legal but Readable)

**Expected Results:**
- Defines indirect/consequential damages  
- Includes 20% liability cap  
- Lists excluded liabilities  
- States calculation basis  
- Approx. 7 AI processing calls  

---

### **Test Case 2: Liquidated Damages (With Documents)**

**Objective:**  
Establish a liquidated damages provision with daily calculation, capped at a 24% annual rate, and clear payment deadlines.

**Jurisdiction:**  
England and Wales

**Uploaded Document:**  
[`liquidated_damages_reference.txt`](./liquidated_damages_reference.txt)

**Reference Highlights:**  
- Damages must be a **genuine pre-estimate of loss**  
- Cannot be **punitive**  
- Must include **clear triggers, methodology, and maximum caps**:contentReference[oaicite:0]{index=0}

**Drafting Style:**  
Legal Formal

**Expected Results:**  
- Uses formal UK legal style  
- References uploaded document principles  
- Contains daily rate formula and annual cap  
- ~9 AI processing calls  

---

### **Test Case 3: Confidentiality Obligations (Complex Test)**

**Objective:**  
Define scope, obligations, exceptions, duration (3 years post-termination), and remedies (including injunctive relief).

**Jurisdiction:**  
Singapore

**Uploaded Documents:**  
- [`confidentiality_guide.txt`](./confidentiality_guide.txt)  
- [`singapore_law_notes.txt`](./singapore_law_notes.txt)

**Reference Highlights:**  
- Confidential information includes **business, technical, and financial data**:contentReference[oaicite:1]{index=1}  
- Singapore law requires **clear definition, reasonable scope, and duration**:contentReference[oaicite:2]{index=2}  
- Courts grant **injunctions for threatened disclosure or ongoing breaches**:contentReference[oaicite:3]{index=3}

**Drafting Style:**  
Balanced (Legal but Readable)

**Expected Results:**  
- 3-year confidentiality period post-termination  
- Standard exceptions list  
- Explicit injunctive relief clause  
- ~10 AI processing calls  

---

### **Test Case 4: Confidentiality + Liquidated Damages Hybrid (Demonstration Extension)**

**Objective:**  
Combine confidentiality and liquidated damages provisions—apply damages of up to 10% of contract value for breach of confidentiality.

**Jurisdiction:**  
Singapore

**Uploaded Documents:**  
- [`confidentiality_guide.txt`](./confidentiality_guide.txt)  
- [`liquidated_damages_reference.txt`](./liquidated_damages_reference.txt)

**Drafting Style:**  
Legal Formal

**Expected Results:**  
- Integrates confidentiality definitions and breach consequences  
- Includes enforceability criteria (genuine pre-estimate of loss)  
- Adds injunctive relief alongside liquidated damages cap  
- ~11 AI processing calls  
