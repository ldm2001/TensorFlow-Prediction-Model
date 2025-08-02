document.addEventListener("DOMContentLoaded", () => {

  const btn_search = document.querySelector("#btn_search")
  btn_search.addEventListener("click", () => {
    const sf = document.querySelector("#sf")
    if(sf.value == '') {
      alert('검색어를 입력해 주세요.')
      sf.focus()
      return false
    }

    const sn = document.querySelector("#sn")

    self.location.href='./member.php?sn=' + sn.value + '&sf=' + sf.value;
  })

  const btn_all = document.querySelector("#btn_all")
  btn_all.addEventListener("click", () => {
    self.location.href = './member.php';
  })

   // 엑셀로 저장
   const btn_excel = document.querySelector("#btn_excel")
   btn_excel.addEventListener("click", () => {
     self.location.href='./member_to_excel.php'
  })

   // 수정 버튼 
   const btn_mem_edit = document.querySelectorAll(".btn_mem_edit") 
   btn_mem_edit.forEach( (box) => {
     box.addEventListener("click", () => {
       const idx = box.dataset.idx
       self.location.href='./member_edit.php?idx=' + idx
     })
   })

    // 삭제 버튼
    const btn_mem_deletes = document.querySelectorAll(".btn_mem_delete")
    btn_mem_deletes.forEach( (box) => {
    box.addEventListener("click", () => {
      if(confirm('본 회원을 삭제하시겠습니까?')) {

        const idx = box.dataset.idx
        const xhr = new XMLHttpRequest()

        const f = new FormData()
        f.append("idx", idx) 

        xhr.open("POST", "./member_del.php", "true")
        xhr.send(f)

        xhr.onload = () => {
          if(xhr.status == 200) {
            const data = JSON.parse(xhr.responseText)
            if(data.result == 'success') {
              self.location.reload()
            }
          } else {
            alert('삭제 실패하였습니다.')
          }
        }
      }else {
      alert('삭제취소')
      }
    })
  })
})