- O:原始
- P:改进
- T:思考
---
>  在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
- O:
```
    public boolean Find(int target, int [][] array) {
        for(int [] cells : array){
            for(int cell : cells){
                if (target == cell){
                    return true;
                }
            }
        }
        return false;
    }
```
- P1:
```
	public boolean Find(int target, int [][] array) {
        int rows = array.length;
        int cols = array[0].length;
        int i = rows - 1, j = 0;//左下角元素坐标
        while (i >= 0 && j < cols) {//使其不超出数组范围
            if (target < array[i][j]) {
                i--;//查找的元素较少，往上找
            } else if (target > array[i][j]) {
                j++;//查找元素较大，往右找
            } else {
                return true;//找到
            }
        }
        return false;
    }
```
---
### 
> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
- O：
```
    public String replaceSpace(StringBuffer str) {
        if (str == null) {
            return null;
        }
        for (int i = 0; i < str.length(); i++) {
            char chr = str.charAt(i);
            if (chr == ' ') {
                str.delete(i, i + 1);
                str.insert(i, "%20");
                i = i + 2;
            }
        }
        return str.toString();
    }
```
---
>输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
- P1(递归):
```
	ArrayList<Integer> arrayList = new ArrayList<>();
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null){
            return arrayList;
        }
        this.printListFromTailToHead(listNode.next);
        arrayList.add(listNode.val);
        return arrayList;
    }
```
---
>输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
- O:
```
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0) {
            return null;
        }
        TreeNode treeNode = new TreeNode(pre[0]);
        for (int i = 0; i < in.length; i++) {
            if (pre[0] == in[i]) {
                // copyOfRange 左闭右开
                treeNode.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                treeNode.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
            }
        }
        return null;
    }
```
---
> 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
- O:
```
Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
     public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        while (!stack2.isEmpty()){
            return stack2.pop();
        }
        while (!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        return stack2.pop();
    }
```
---
> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
- O（封啸自创解法，真牛皮）:
```
public int minNumberInRotateArray(int[] array) {
        if (array.length == 0) {
            return 0;
        }
        int len = array.length;
        int low = 0;
        int high = len - 1;
        while (low < high) {
            int mid = low + high >> 1;
            if (array[mid] <= array[len - 1]) {//如果左侧小于右侧，
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return array[low];
    }
```
- T:
1. 不一定非要考虑题中的旋转。通过求出最小亦是求出答案。
通过观察可知。如果array[mid]<=array[high] 可知，如果成立，那么符合二分查找。数据最小的一定在左边。其他情况。说明不成立。那么范围一定得从高位中找。
2. 
正数：r = 20 << 2;结果：r = 80

负数：r = -20 << 2;结果：r = -80

正数：r = 20 >> 2;结果：r = 5

负数：r = -20 >> 2;结果：r = -5
> 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
- O:
```
	public int Fibonacci(int n) {
        if(n == 0){
            return 0;
        }else if(n == 1 ||n == 2){
            return 1;
        }else{
            return Fibonacci(n-1)+Fibonacci(n-2);
        }
    }
```
- P:
```
 	public int Fibonacci(int n) {
        if(n <= 1){
            return n;
        }
        int count = 0;
        int tempOne = 0;
        int tempTwo = 1;
        for(int i = 2;i<= n;i++){
            count = tempOne + tempTwo;
            tempOne = tempTwo;
            tempTwo = count;
        }
        return count;
    }
```
- T:
1.  斐波那契数列
            1                                x = 1
f(x) =   1       x = 2
           f(x - 1)  + f(x - 2)        x >= 3
> 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
- O:
```
	public int JumpFloor(int target) {
        if(target<=0){
            return 0;
        }
        if(target == 1){
            return 1;
        } 
        if(target == 2){
            return 2;
        } 
        return JumpFloor(target-1)+JumpFloor(target-2);
    }
```
- P:
```
```
- T: 