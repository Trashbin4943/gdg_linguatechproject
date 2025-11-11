from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

'''
STT로 음성의 텍스트화 --> 모델에서 감정 학습 --> 상담사 응답을 생성하는 함수입니다.
KoGPT를 사용해 감정 라벨과 사용자 발화를 기반으로 상담 응답을 생성합니다.
'''

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

'''
STT로 음성을 텍스트화 → 감정 분석 → KoGPT로 민원 응답 생성
민원 상담에 맞게 감정별 말투와 예시 응답을 포함한 프롬프트를 구성합니다.
'''

def generate_response(emotion_label, user_text):
    # 감정 라벨 매핑
    emotion_map = {0: "불만", 1: "분노", 2: "불안", 3: "중립", 4: "감사", 5: "요청", 6: "혼란"}

    # 감정별 말투 스타일
    style_map = {
        "불만": "공감하며 사과하고 해결 방안을 안내하는 말투로",
        "분노": "책임감 있게 사과하고 신속한 조치를 약속하는 말투로",
        "불안": "안심시키고 절차를 명확히 설명하는 말투로",
        "중립": "정중하고 간결하게 안내하는 말투로",
        "감사": "감사 인사를 공손하게 전달하는 말투로",
        "요청": "요청 사항을 확인하고 처리 절차를 안내하는 말투로",
        "혼란": "상황을 정리하고 명확하게 설명하는 말투로"
    }

    # 감정별 예시 응답
    example_map = {
        "불만": "불편을 드려 죄송합니다. 해당 내용은 즉시 확인 후 처리하겠습니다.",
        "분노": "심려를 끼쳐드려 죄송합니다. 빠르게 조치하겠습니다.",
        "불안": "걱정되실 수 있는 상황입니다. 정확한 절차를 안내드릴게요.",
        "중립": "문의하신 내용은 아래 절차에 따라 처리됩니다.",
        "감사": "소중한 의견 감사합니다. 더욱 노력하겠습니다.",
        "요청": "요청하신 사항은 확인 후 처리 절차를 안내드리겠습니다.",
        "혼란": "혼란을 드려 죄송합니다. 상황을 정리해 설명드리겠습니다."
    }

    # 감정 이름 추출
    emotion = emotion_map.get(emotion_label, "중립")
    style = style_map.get(emotion, "정중하고 간결한 말투로")
    example = example_map.get(emotion, "")

    # 프롬프트 구성
    prompt = f"""민원 상담 응답 생성기
    사용자 감정: {emotion}
    사용자 발화: {user_text}
    상담사 응답 스타일: {style}
    상담사 응답 예시: {example}
    상담사 응답:"""

    # KoGPT 모델 로딩 및 응답 생성
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)