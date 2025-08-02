function getUrlParams() {     
  const params = {};  
  window.location.search.replace(/[?&]+([^=&]+)=([^&]*)/gi, 
    function(str, key, value) { 
        params[key] = value; 
      }
  );     
  return params; 
}

function getExtensionOfFilename(filename) {
  const filelen = filename.length; // 문자열의 길이
  const lastdot = filename.lastIndexOf('.');
  return filename.substring(lastdot + 1, filelen).toLowerCase();
}

document.addEventListener("DOMContentLoaded", () => {

  // 게시판 목록으로 이동하기
  const btn_board_list = document.querySelector("#btn_board_list")
  btn_board_list.addEventListener("click", () => {

    const params = getUrlParams()
    self.location.href='./board.php?bcode=' + params['bcode']
  })

  // 게시물 작성후 확인 버튼 클릭시
  const btn_write_submit = document.querySelector("#btn_write_submit")
  btn_write_submit.addEventListener("click", () => {
    const id_subject = document.querySelector("#id_subject");
    if(id_subject.value == '') {
      alert('게시물 제목을 입력해 주세요')
      id_subject.focus()
      return false  
    }
    
    const markupStr = $('#summernote').summernote('code')
    if(markupStr == '<p><br></p>') {
      alert('내용을 입력하세요.')
      return false
    }

    // 파일 첨부
    const id_attach = document.querySelector("#id_attach")
    //const file = id_attach.files[0]

    if(id_attach.files.length > 3) {
      alert('첨부할 파일 수는 3개까지입니다.')
      id_attach.value = ''
      return false
    }

    const params = getUrlParams()

    const f = new FormData()
    f.append("subject", id_subject.value) // 게시물 제목
    f.append("content", markupStr) // 게시물 내용
    f.append("bcode", params['bcode']) // 게시판 코드
    f.append("mode", "input") // 모드 : 글등록
    //f.append("files", file) // 파일

    let ext = ''

    for(const file of id_attach.files) {
      if(file.size > 40 * 1024 * 1024) {
        alert('파일 용량이 40메가보다 큰 파일이 첨부되었습니다. 확인 바랍니다.')
        id_attach.value = ''
        return false
      }

      ext = getExtensionOfFilename(file.name);
      if(ext == 'txt' || ext == 'exe' || ext == 'xls' || ext == 'php' || ext == 'js') {
        alert('첨부할 수 없는 포멧의 파일이 첨부되었습니다.(exe, txt, php, js ..)')
        id_attach.value = ''
        return false
      }

      f.append("files[]", file) // 파일
    }
 
    const xhr = new XMLHttpRequest()
    xhr.open("post", "./pg/board_process.php", true)
    xhr.send(f)

    xhr.onload = () => {
      if(xhr.status == 200) {

        const data = JSON.parse(xhr.responseText)
        if(data.result == 'success') {
          alert('글 등록이 성공했습니다.')
          self.location.href='./board.php?bcode=' + params['bcode']
        }else if(data.result == 'file_upload_count_exeed') {
          alert('파일 업로드 갯수를 초과했습니다.')
          id_attach.value = ''
          return false
        }else if(data.result == 'post_size_exceed') {
          alert('첨부파일의 용량이 큽니다. 작은 파일로 첨부해 주세요.')
          id_attach.value = ''
          return false
        }else if(data.result == 'not_allowed_file') {
          alert('첨부할 수 없는 포멧의 파일이 첨부되었습니다.(exe, txt ..)')
          id_attach.value = ''
          return false
        }

      } else if(xhr.status == 404) {
        alert('통신실패, 파일없습니다.')
      }
    }

  })

  const id_attach = document.querySelector("#id_attach")
  id_attach.addEventListener("change", () => {
    if(id_attach.files.length > 3) {
      alert('파일은 3개까지만 첨부가 가능합니다')
      id_attach.value = ''
      return false
    }
    
  })

})