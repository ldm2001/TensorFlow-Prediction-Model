<?php

$g_title = '농수산물 장터';
$js_array = ['js/home.js'];

$menu_code = 'home';

include 'inc/common.php';
include 'inc/dbconfig.php';

include 'inc/boardmanage.php';
$boardm = new BoardManage($db);
$boardArr = $boardm->list();

include 'inc_header.php';
?>

<main class="border rounded-2 p-5 d-flex gap-5" style="height: calc(100vh - 257px)">
  <img src="images/사과 소매가0821.svg" style="max-width: 67%; height: auto;" alt="사과 소매 가격 그래프">
  
  <div class="table-responsive" style="flex: 1;">
    <table class="table table-bordered table-striped" style="width: 100%; font-size: 1.2rem; text-align: center;">
      <thead>
        <tr>
          <th style="padding: 12px;">날짜</th>
          <th style="padding: 12px;">가격</th>
          <th style="padding: 12px;">등락률</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding: 12px;">24.08.21</td>
          <td style="padding: 12px;">32,575</td>
          <td style="padding: 12px;">6.33</td>
        </tr>
        <tr>
          <td style="padding: 12px;">24.08.20</td>
          <td style="padding: 12px;">30,636</td>
          <td style="padding: 12px;">-7.69</td>
        </tr>
        <tr>
          <td style="padding: 12px;">24.08.19</td>
          <td style="padding: 12px;">33,189</td>
          <td style="padding: 12px;">7.25</td>
        </tr>
          <tr>
          <td style="padding: 12px;">24.08.16</td>
          <td style="padding: 12px;">30,946</td>
          <td style="padding: 12px;">10.47</td>
        </tr>
        <tr>
          <td style="padding: 12px;">24.08.14</td>
          <td style="padding: 12px;">28,014</td>
          <td style="padding: 12px;">0.05</td>
        </tr>
        <tr>
          <td style="padding: 12px;">24.08.13</td>
          <td style="padding: 12px;">28,000</td>
          <td style="padding: 12px;">-0.4</td>
        </tr>
        <tr>
          <td style="padding: 12px;">24.08.12</td>
          <td style="padding: 12px;">28,112</td>
          <td style="padding: 12px;">-3.73</td>
        </tr>
        <tr>
          <td style="padding: 12px;">24.08.09</td>
          <td style="padding: 12px;">29,200</td>
          <td style="padding: 12px;">1.15</td>
        </tr>
      </tbody>
    </table>
  </div>
</main>

<?php
include 'inc_footer.php';
?>
