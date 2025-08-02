<?php
$g_title = '회원가입을 축하드립니다!';
$js_array = ['js/member_success.js'];

$menu_code = 'member';

include 'inc_header.php';
?>

<main class="w-75 mx-auto border rounded-5 p-5 d-flex gap-5">

<img src="images/apple logo.jpg" alt="">
<div>
  <h3>축하드립니다! 가입 완료되었습니다.</h3>
  <p>환영합니다 회원님! 좋은 하루 보내세요.</p>
  <button class="btn btn-primary" id="btn_login">로그인 하기</button>
</div>
</main>

<?php
include 'inc_footer.php';
?>