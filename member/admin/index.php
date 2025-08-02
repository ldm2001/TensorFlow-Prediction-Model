<?php

$g_title = '사과 장터';
$js_array = ['js/home.js'];

$menu_code = 'home';

include 'inc_common.php';
include 'inc_header.php';

?>
<main class="w-75 mx-auto border rounded-5 p-5 d-flex gap-5" style="height: calc(100vh - 257px)">

  <div class="mx-auto p-5">
    <h1 class="text-center">Warning!</h1>
    <h2 class="text-center">이곳은 관리자 페이지 입니다.</h2>
  </div>
</main>

<?php
include 'inc_footer.php';
?>