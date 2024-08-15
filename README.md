# 📊기업맞춤형 AI면접 플랫폼 AIT
> **KT AIVLE SCHOOL 5기 11조 빅프로젝트**<br/> **개발기간: 2024.07**

## 🔗Link(시연영상 링크)
 - https://youtu.be/8Z8Ex54-4CM

## 👩‍💻AI Developers

|               유희권               |               김진혁               |               김가린               | 
| :---------------------------------: | :-------------------------------------: | :-------------------------------------: |
| [HuiGwon Ryu](https://github.com/AnthonyRyu) | [Aivle-noin](https://github.com/Aivle-noin) | [kimcookie00](https://github.com/kimcookie00) |
<br>

## 문맥 분석 서버
LLM과 BERT-base를 활용해 기업 공통 질문과 자소서 기반 질문을 분석
- GPT-4o(40%), Emotion Score(20%), LSA Score(20%), Similarity Score(20%)

### 환경
```
Python == 3.11.9    
numpy == 1.26.4   
Flask == 3.0.3   
scikit-learn (sklearn) == 1.5.0   
openai == 1.35.10 (에러날 시 확인 필요)   
torch == 2.3.1   
transformers == 4.42.3
```

### 사용법

- Input
    ```Json
    {
        "occupation": "IT",
        "question": "KT 인재상 중에 본인에게 적합하다고 생각하는것과 이유를 설명해주세요",
        "answer": "안녕하세요. KT 인재상 중에서 저에게 가장 적합하다고 생각하는 부분은 도전 정신입니다. 저는 항상 새로운 도전과 목표를 설정하며 성장을 추구하는 사람입니다. 예를 들어, 대학 시절에는 학업과 병행하여 다양한 프로젝트와 인턴십에 참여하면서 실무 경험을 쌓았습니다. 이러한 경험들은 저의 문제 해결 능력과 창의성을 크게 향상시켜 주었고, 이는 곧 KT의 혁신적인 사업 환경에서도 큰 도움이 될 것이라고 생각합니다. 또한, 저는 변화에 빠르게 적응하며 새로운 기술과 트렌드를 학습하는 것을 즐깁니다. KT는 빠르게 변화하는 ICT 산업의 선두주자로서 지속적인 혁신과 발전을 추구하는 기업입니다. 따라서 저의 도전 정신과 학습 능력은 KT의 목표와 매우 부합한다고 생각합니다. 마지막으로, 도전 정신은 팀워크와 협업에서도 중요한 요소라고 생각합니다. 저는 다양한 팀 프로젝트를 통해 협력하고 소통하는 능력을 키워왔으며, 이를 바탕으로 KT의 다양한 부서와 협력하여 성공적인 결과를 만들어낼 자신이 있습니다. 감사합니다."
    }
    ```
- Output
    ```Json
    {
        "context_score": 78.36979951789213,
        "emotion_score": 5.469755530357361,
        "lsa_score": 25,
        "munmek": {
            "Clarity": "답변이 명확하고 쉽게 이해될 수 있도록 잘 구성되었으며, 지원자의 도전 정신과 이를 뒷받침하는 사례가 명료하게 제시되었습니다. 다만 명확성을 더 높이려면 구체적인 프로젝트 이름이나 인턴십 세부 내용을 추가하는 것이 바람직합니다.",
            "Logicality": "지원자가 자신이 가진 도전 정신이 KT의 목표와 어떻게 일치하는지를 논리적으로 설명했습니다. 대학 시절의 경험과 결과를 구체적으로 제시하여 논리적인 연관성을 높였으나, 더 다양한 사례가 있었다면 더 높은 점수를 줄 수 있었을 것입니다.",
            "Question Comprehension": "질문에 대해 정확히 이해하고 답변을 구성했으며, KT가 추구하는 인재상과 자신의 특성을 연결지어 설명한 점이 매우 좋았습니다. 다만, 약간의 추가 정보가 제공되었다면 최상의 점수를 받을 수 있을 것입니다.",
            "Relevance": "지원자가 KT 인재상 중 '도전 정신'을 선택하고 자신의 성향과 경험을 구체적으로 설명하였습니다. KT의 혁신적이고 빠르게 변화하는 환경에 잘 맞는다고 평가했습니다. 점수를 내린 이유는 다루지 않은 다른 인재상의 요소가 있을 수 있다는 점입니다.",
            "munmek_score": 37.400000000000006
        },
        "munmek_score": 37.400000000000006,
        "similarity_score": 10.500043987534768
    }
    ```
