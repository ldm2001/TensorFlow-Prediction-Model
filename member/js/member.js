document.addEventListener("DOMContentLoaded", () => {
  const btn_member = document.querySelector("#btn_member")

  btn_member.addEventListener("click", () => {
    const chk_member1 = document.querySelector("#chk_member1")
    if(chk_member1.checked !== true) {
      alert('회원 약관에 동의해주시기 바랍니다.')
      return false
    }

    const chk_member2 = document.querySelector("#chk_member2")
    if(chk_member2.checked !== true) {
      alert('개인정보 취급 방침에 동의해주시기 바랍니다.')
      return false
    }

    const f= document.stipulation_form
    f.chk.value = 1
    f.submit()
  })
})