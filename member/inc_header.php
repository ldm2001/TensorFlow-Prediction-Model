<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title><?= (isset($g_title) && $g_title != '') ? $g_title : '사과장터'; ?></title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-aFq/bzH65dt+w6FI2ooMVUpc+21e0SRygnTpmBvdBgSdnuTN7QbdgL+OapgHtvPp" crossorigin="anonymous">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha2/dist/js/bootstrap.bundle.min.js" integrity="sha384-qKXV1j0HvMUeCBQ+QVp7JcfGl760yU08IQ+GpUo5hlbpg51QRiuqHAJz8+BrxE/N" crossorigin="anonymous"></script>
<?php
if(isset($js_array)) {
  foreach($js_array AS $var) {
    echo '<script src="'.$var.'?v='. date('YmdHis') .'"></script>'.PHP_EOL;
  }
}  
?>  

<style>
    .hangul {
      font-family: 'Nanum Barun Gothic' !important;
    }

    .carousel-item img {
      width: 100%;
      height: 500px; /* 슬라이드 높이 설정 */
      object-fit: cover; /* 이미지가 꽉 차게 */
    }

    /* 메뉴 글자색상 변경 */
    .nav-link {
      color: black; /* 원하는 색상으로 변경 */
    }

    .nav-link:hover {
      color: black; /* 호버시 색상 변경 */
    }

    .nav-item{
      position: relative;
      display: inline-block;
    }
    .nav-item::after {
      content: '';
      position: absolute;
      left: 15%;
      bottom: 0;
      width: 0;
      height: 2px; /* 밑줄 두께 */
      background-color: black; /* 밑줄 색상 */
      transition: width 0.5s ease-in-out; /* 애니메이션 시간 및 효과 */
}
    .nav-item:hover::after {
      width: 70%;
    }

    .nav-item:hover{
      font-weight: bold;
      /* text-decoration: underline;
      text-underline-offset: 10px;
      text-decoration-thickness: 2px; */
    }
    .active{
      font-weight: bold;
      text-decoration: underline;
      text-underline-offset: 12px;
      text-decoration-thickness: 2px;
      background-color: white !important;
      color: black !important;
    }
  </style>

  
</head>
<body>
  <div class="container">
    <header class="d-flex flex-wrap py-3 mb-4 border-bottom">
      <a href="/" class="d-flex align-items-center mb-3 mb-md-0 me-2 link-body-emphasis text-decoration-none">
        <img src="images/icon.jpg" style="width:2rem" class="me-2">
        <span class="fs-5 fw-bold">사과장터</span>
      </a>

      <ul class="nav nav-pills">
<?php if(isset($_SESSION['ses_id']) && $_SESSION['ses_id'] != ''){ // 로그인 상태 ?> 
        <li class="nav-item">
          <a href="index.php" class="nav-link <?= ($menu_code == 'home') ? 'active': ''; ?>">시세</a>
        </li>
        <li class="nav-item">
          <a href="company.php" class="nav-link <?= ($menu_code == 'company') ? 'active': ''; ?>">소개</a>
        </li>
        <?php if($_SESSION['ses_level'] == 10) { ?>   
        <li class="nav-item">
          <a href="./admin/" class="nav-link <?= ($menu_code == 'member') ? 'active': ''; ?>">관리자</a>
        </li>
       <?php } else { ?> 
        <li class="nav-item">
          <a href="mypage.php" class="nav-link <?= ($menu_code == 'member') ? 'active': ''; ?>">회원정보</a>
        </li>
       <?php } ?>
       <?php
            foreach($boardArr = isset($boardArr) ? $boardArr : array() AS $row) {
              echo '<li class="nav-item"><a href="board.php?bcode='.$row['bcode'].'" class="nav-link';
              if(isset($_GET['bcode']) && $_GET['bcode'] == $row['bcode']) {
                echo ' active';
              }
              echo '">'.$row['name'].'</a></li>';
            }
       ?>

        <li class="nav-item">
          <a href="./pg/logout.php" class="nav-link <?= ($menu_code == 'login') ? 'active': ''; ?>">로그아웃</a>
        </li>
<?php } else { // 로그인 안된 상태 ?> 
        <li class="nav-item"><a href="index.php" class="nav-link <?= ($menu_code == 'home') ? 'active': ''; ?>">시세</a></li>
        <li class="nav-item"><a href="company.php" class="nav-link <?= ($menu_code == 'company') ? 'active': ''; ?>">소개</a></li>
        <li class="nav-item"><a href="stipulation.php" class="nav-link <?= ($menu_code == 'member') ? 'active': ''; ?>">회원가입</a></li>
        <li class="nav-item"><a href="board.php" class="nav-link <?= ($menu_code == 'board') ? 'active': ''; ?>">게시판</a></li>
        <li class="nav-item"><a href="login.php" class="nav-link <?= ($menu_code == 'login') ? 'active': ''; ?>">로그인</a></li>
<?php
}
?>
      </ul>
    </header>