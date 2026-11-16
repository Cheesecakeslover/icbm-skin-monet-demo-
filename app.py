import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Interactive CBM for Skin Disease Classification",
    page_icon="🩺",
    layout="wide",
)

# คำอธิบายกลุ่มโรคผิวหนังทั้ง 7 คลาส (HAM10000)
CLASS_INFO = {
    "mel": ("Melanoma (มะเร็งผิวหนังเมลาโนมา)", "ความเสี่ยงสูง — ควรส่งตรวจยืนยันทางพยาธิวิทยาโดยด่วน"),
    "nv": ("Melanocytic Nevi (ไฝธรรมดา)", "รอยโรคไม่ร้ายแรง — ติดตามการเปลี่ยนแปลงตามรอบปกติ"),
    "bcc": ("Basal Cell Carcinoma (มะเร็งผิวหนังเบซัลเซลล์)", "ความเสี่ยงสูง — มะเร็งผิวหนังชนิดพบบ่อย ควรพบแพทย์เฉพาะทาง"),
    "akiec": ("Actinic Keratoses (รอยโรคก่อนมะเร็งผิวหนัง)", "ความเสี่ยงปานกลาง-สูง — ภาวะก่อนเป็นมะเร็งจากแสงแดด"),
    "bkl": ("Benign Keratosis (กระเนื้อ / ติ่งเนื้อไม่ร้ายแรง)", "รอยโรคไม่ร้ายแรง — มักพบตามวัย"),
    "df": ("Dermatofibroma (เนื้องอกเส้นใยผิวหนัง)", "รอยโรคไม่ร้ายแรง — ก้อนนูนแข็งใต้ผิวหนัง"),
    "vasc": ("Vascular Lesions (รอยโรคหลอดเลือด)", "รอยโรคไม่ร้ายแรง — ความผิดปกติของหลอดเลือดฝอย"),
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

@st.cache_resource
def load_monet_icbm(model_path="icbm_skin_monet.joblib"):
    device = get_device()
    bundle = joblib.load(model_path)
    monet_id = "suinleelab/monet"

    vision_model = CLIPModel.from_pretrained(monet_id).float().to(device).eval()
    try:
        processor = CLIPProcessor.from_pretrained(monet_id)
    except Exception:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    return bundle, vision_model, processor, device

def extract_concept_scores(image: Image.Image, bundle, vision_model, processor, device):
    px = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    px = px.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        out = vision_model.get_image_features(pixel_values=px)
        feat = out if torch.is_tensor(out) else getattr(out, "image_embeds", getattr(out, "pooler_output", None))
        feat = feat / torch.sqrt(torch.sum(feat * feat, dim=-1, keepdim=True)).clamp_min(1e-12)
        x_vec = feat.cpu().numpy()

    T_conc = np.asarray(bundle["T_conc"], dtype=np.float32)
    mu = np.asarray(bundle["mu"], dtype=np.float32)
    sd = np.asarray(bundle["sd"], dtype=np.float32)

    c_raw = x_vec @ T_conc.T
    return ((c_raw - mu) / np.maximum(sd, 1e-12))[0]

# ==============================================================================
# ส่วนหัวของเว็บและกรอบแนวคิดวิจัย
# ==============================================================================
st.title("🩺 Interactive Concept Bottleneck Models for Skin Disease Classification")
st.markdown(
    "ระบบวินิจฉัยโรคผิวหนังแบบโปร่งใสด้วย **MONET** ที่เปิดโอกาสให้ผู้เชี่ยวชาญตรวจสอบและแก้ไขลักษณะอาการทางคลินิกได้"
)

with st.expander("📌 ที่มาและความสำคัญ / โจทย์วิจัย (คลิกเพื่อดูรายละเอียด)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **ที่มาและความสำคัญ:**
            ปัจจุบันโมเดล Deep Learning ส่วนใหญ่ในการวินิจฉัยโรคผิวหนังทำงานเป็น **"กล่องดำ" (Black Box)**
            ซึ่งแพทย์ไม่สามารถตรวจสอบเหตุผลที่แท้จริงของการตัดสินใจได้ แม้จะมีความพยายามใช้ **Concept Bottleneck Models (CBM)**
            เพื่ออธิบายผลโดยการดึงอาการทางแพทย์ (เช่น รอยแดง, ขุย) ออกมา แต่หากระบบสกัดอาการผิดพลาดตั้งแต่ต้นทาง
            (เช่น มองไม่เห็นขุยสะเก็ด) จะทำให้โมเดลปลายทางทำนายโรคผิดพลาดตามไปด้วยโดยที่ผู้ใช้แก้ไขไม่ได้
            """
        )
    with col_b:
        st.markdown(
            """
            **โจทย์วิจัย:**
            *ระบบจำแนกโรคผิวหนังที่เปิดโอกาสให้ผู้เชี่ยวชาญสามารถตรวจสอบและแก้ไขค่าอาการ
            สามารถช่วยเพิ่มความแม่นยำและความน่าเชื่อถือในการวินิจฉัยโรคเมื่อเปรียบเทียบกับ CBM แบบดั้งเดิมหรือไม่?*
            - **Concept Extractor:** MONET (`suinleelab/monet`)
            - **Interactive Layer:** Expert-in-the-Loop Concept Intervention
            """
        )

# ตรวจสอบไฟล์โมเดล
MODEL_FILE = "icbm_skin_monet.joblib"
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ ไม่พบไฟล์ `{MODEL_FILE}` ในโฟลเดอร์ปัจจุบัน กรุณานำไฟล์มาวางตำแหน่งเดียวกับ `app.py` แล้วรีเฟรชหน้าเว็บ")
    st.stop()

with st.spinner("กำลังโหลดโมเดล MONET และตัวจำแนกโรค ICBM..."):
    bundle, vision_model, processor, device = load_monet_icbm(MODEL_FILE)

cbm = bundle["cbm"]
classes = list(bundle["classes"])
concepts = list(bundle["concepts"])

# ==============================================================================
# แถบด้านข้าง (Sidebar)
# ==============================================================================
with st.sidebar:
    st.header("ℹ️ ข้อมูลระบบ")
    st.success(f"โมเดล: **MONET + ICBM**\n\nอุปกรณ์ประมวลผล: **{device}**\n\nจำนวนอาการที่ตรวจจับ: **{len(concepts)} Concepts**")
    st.divider()
    st.markdown("**วิธีอ่านค่าคะแนนอาการ (Z-score):**")
    st.markdown(
        """
        - **+1.0 ถึง +3.5** : ตรวจพบอาการชัดเจน
        - **-0.5 ถึง +0.5** : ไม่ชัดเจน / ก้ำกึ่ง
        - **-3.5 ถึง -1.0** : ไม่พบลักษณะอาการนี้
        """
    )

# ==============================================================================
# พื้นที่ทำงานหลัก: อัปโหลดภาพ -> ตรวจสอบและแก้ไขอาการ -> ดูผลวินิจฉัย
# ==============================================================================
uploaded = st.file_uploader("📤 อัปโหลดภาพถ่ายรอยโรคผิวหนัง (JPG / PNG) เพื่อเริ่มการวินิจฉัย", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    file_sig = f"{uploaded.name}_{uploaded.size}"

    # สกัดค่า Concept จาก MONET เมื่ออัปโหลดรูปใหม่
    if st.session_state.get("file_sig") != file_sig:
        z_ai = extract_concept_scores(img, bundle, vision_model, processor, device)
        st.session_state["file_sig"] = file_sig
        st.session_state["z_ai"] = z_ai
        st.session_state["z_base"] = z_ai.copy()
        st.session_state["is_corrupted"] = False
        for i, val in enumerate(z_ai):
            st.session_state[f"c_{i}"] = float(np.clip(val, -3.5, 3.5))

    z_ai = st.session_state["z_ai"]
    z_base = st.session_state["z_base"]

    # ปุ่มจำลองสถานการณ์ตามโจทย์วิจัย
    st.markdown("#### 🧪 จำลองสถานการณ์ตามโจทย์วิจัย:")
    btn_col1, btn_col2, btn_col3 = st.columns([1.3, 1.2, 1.8])
    with btn_col1:
        if st.button("❌ จำลองระบบสกัดอาการผิดพลาด (เช่น มองไม่เห็นขุย)", use_container_width=True):
            rng = np.random.RandomState(42)
            z_corrupted = z_ai.copy()
            # สุ่มรบกวน 30% ของอาการ และกดค่าอาการเด่นให้ลดลงเหมือนระบบมองไม่เห็นอาการสำคัญ
            top_pos = np.argsort(-z_corrupted)[:3]
            z_corrupted[top_pos] = -1.5
            noise_mask = rng.rand(len(z_corrupted)) < 0.25
            z_corrupted[noise_mask] = rng.randn(int(noise_mask.sum())) * 1.2
            st.session_state["z_base"] = z_corrupted
            st.session_state["is_corrupted"] = True
            for i, val in enumerate(z_corrupted):
                st.session_state[f"c_{i}"] = float(np.clip(val, -3.5, 3.5))
            st.rerun()
    with btn_col2:
        if st.button("🔄 รีเซ็ตกลับเป็นค่าเริ่มต้นของ AI", use_container_width=True):
            st.session_state["z_base"] = z_ai.copy()
            st.session_state["is_corrupted"] = False
            for i, val in enumerate(z_ai):
                st.session_state[f"c_{i}"] = float(np.clip(val, -3.5, 3.5))
            st.rerun()
    with btn_col3:
        if st.session_state.get("is_corrupted"):
            st.warning("⚠️ กำลังจำลองกรณี AI สกัดอาการต้นทางผิดพลาด — ลองแก้ไขค่าอาการฝั่งขวาเพื่อกู้คืนคำวินิจฉัย!")

    st.divider()
    col_left, col_right = st.columns([1.05, 1.35], gap="large")

    # --- ฝั่งขวา: ให้ผู้เชี่ยวชาญตรวจสอบและแก้ไขค่า Concept ---
    with col_right:
        st.subheader("🔍 2. ตรวจสอบและแก้ไขค่าอาการ (Concept Intervention)")
        show_all = st.checkbox("แสดงลักษณะอาการ (Concepts) ทั้งหมด", value=False)

        # เรียงอาการที่เด่นชัดที่สุดขึ้นมาก่อน
        order = np.argsort(-np.abs(z_base))
        display_indices = order if show_all else order[:8]

        edited_indices = []
        z_intervened = z_base.copy()

        for i in display_indices:
            base_val = float(np.clip(z_base[i], -3.5, 3.5))
            badge = "🟢 พบอาการ" if base_val > 0.6 else ("🔴 ไม่พบ" if base_val < -0.6 else "⚪ ก้ำกึ่ง")
            cur_val = st.slider(
                label=f"{concepts[i]}  |  ค่าจากระบบ: {base_val:+.2f} ({badge})",
                min_value=-3.5,
                max_value=3.5,
                value=st.session_state[f"c_{i}"],
                step=0.1,
                key=f"c_{i}",
            )
            z_intervened[i] = cur_val
            if abs(cur_val - base_val) > 1e-3:
                edited_indices.append(i)

    # --- ฝั่งซ้าย: เปรียบเทียบผล Traditional CBM vs Interactive CBM ---
    with col_left:
        st.subheader("🖼️ 1. ภาพรอยโรคและผลวินิจฉัยเปรียบเทียบ")
        st.image(img, caption=f"ภาพที่อัปโหลด: {uploaded.name}", use_container_width=True)

        # คำนวณความน่าจะเป็นทั้ง 2 แบบ
        prob_trad = cbm.predict_proba(z_base.reshape(1, -1))[0]
        prob_icbm = cbm.predict_proba(z_intervened.reshape(1, -1))[0]

        idx_trad = int(np.argmax(prob_trad))
        idx_icbm = int(np.argmax(prob_icbm))

        code_trad, code_icbm = classes[idx_trad], classes[idx_icbm]
        name_trad, note_trad = CLASS_INFO.get(code_trad, (code_trad, ""))
        name_icbm, note_icbm = CLASS_INFO.get(code_icbm, (code_icbm, ""))

        # ผลลัพธ์ Traditional CBM
        st.markdown("##### 1️⃣ Traditional CBM (ระบบอัตโนมัติเดิม — แก้ไขไม่ได้)")
        st.info(f"**ผลวินิจฉัย:** {name_trad}\n\nความมั่นใจ: **{prob_trad[idx_trad]*100:.2f}%** | *{note_trad}*")

        # ผลลัพธ์ Interactive CBM
        st.markdown(f"##### 2️⃣ Interactive CBM (หลังผู้เชี่ยวชาญแก้ไข {len(edited_indices)} อาการ)")
        if len(edited_indices) == 0:
            st.caption("👈 *เลื่อนแถบปรับแก้อาการฝั่งขวาเพื่อดูผลลัพธ์ของ Interactive CBM*")
        else:
            diff_conf = (prob_icbm[idx_icbm] - prob_trad[idx_icbm]) * 100
            st.success(
                f"**ผลวินิจฉัยใหม่:** {name_icbm}\n\n"
                f"ความมั่นใจ: **{prob_icbm[idx_icbm]*100:.2f}%** ({diff_conf:+.2f}%) | *{note_icbm}*"
            )

        # ตารางและกราฟแท่งเปรียบเทียบ
        df_comp = pd.DataFrame({
            "กลุ่มโรคผิวหนัง": [CLASS_INFO.get(c, (c, ""))[0] for c in classes],
            "Traditional CBM (%)": np.round(prob_trad * 100, 2),
            "Interactive CBM (%)": np.round(prob_icbm * 100, 2),
        }).sort_values("Interactive CBM (%)", ascending=False)

        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        st.bar_chart(df_comp.set_index("กลุ่มโรคผิวหนัง")[["Traditional CBM (%)", "Interactive CBM (%)"]])
else:
    st.info("👆 กรุณาอัปโหลดภาพถ่ายรอยโรคผิวหนังด้านบนเพื่อเริ่มการวินิจฉัย")