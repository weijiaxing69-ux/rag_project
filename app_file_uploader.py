import streamlit as st
st.title("知识库更新")

uploader_file=st.file_uploader(
    label="请上传文件",
    type=['txt'],
    accept_multiple_files=False
)

if uploader_file is not None:
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024

    st.subheader(f"文件名: {file_name}")
    st.write(f"格式: {file_type} | 大小: {file_size}")

   